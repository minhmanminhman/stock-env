import numpy as np
from abc import ABC, abstractmethod

import torch as th
from stock_env.algos.buffer import RolloutBuffer
from stock_env.common.env_utils import get_device


class BasePPO(ABC):
    def __init__(self, args, envs, agent):
        self.args = args
        self.envs = envs
        self.agent = agent
        self.device = get_device()
        self.buffer = RolloutBuffer(
            num_steps=self.args.num_steps,
            envs=self.envs,
            device=self.device,
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
        )
        self.global_step = 0

    @abstractmethod
    def learn(self, *arg, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _train(self, *arg, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def _env_reset(self, envs=None):
        if envs is None:
            envs = self.envs
        self._obs, self._infos = envs.reset()
        self._dones = th.zeros((self.args.num_envs,))
        self._obs = th.Tensor(self._obs).to(self.device)

    def _collect_rollout(self, agent, callback, envs=None):
        assert self._obs is not None, "Please call '_env_reset' first"
        if envs is None:
            envs = self.envs
        self.buffer.reset()
        for step in range(self.buffer.num_steps):
            self.global_step += 1 * self.args.num_envs
            with th.no_grad():
                actions, values, log_probs, _ = agent.get_action_and_value(self._obs)
                values = values.flatten()

            (
                next_obs,
                rewards,
                next_terminated,
                next_truncated,
                next_infos,
            ) = envs.step(actions.cpu().numpy())
            next_obs = th.Tensor(next_obs).to(self.device)
            rewards = th.Tensor(rewards).to(self.device).flatten()
            next_dones = np.logical_or(next_terminated, next_truncated)
            next_dones = th.Tensor(next_dones).to(self.device).flatten()

            if any(next_dones):
                # find which envs are done
                idx = np.where(next_dones)[0]
                # Handle timeout by bootstraping with value function
                # NOTES: for timeout env
                if self.args.is_timeout:
                    final_observation = next_infos["final_observation"][idx][0]
                    # calculated value of final observation
                    with th.no_grad():
                        final_value = agent.get_value(
                            th.Tensor(final_observation).to(self.device)
                        )
                    rewards[idx] += self.args.gamma * final_value

                # logging
                final_infos = next_infos["final_info"][idx]
                mean_reward = np.mean(
                    [
                        info["episode"]["r"]
                        for info in final_infos
                        if "episode" in info.keys()
                    ]
                )

                if callback is not None:
                    callback.update_locals(locals(), globals())
                    callback.on_dones()

            # add to buffer
            self.buffer.add(
                index=step,
                obs=self._obs,
                actions=actions,
                logprobs=log_probs,
                rewards=rewards,
                dones=self._dones,
                values=values,
            )
            self._obs = next_obs
            self._dones = next_dones

        # compute returns
        with th.no_grad():
            next_value = agent.get_value(next_obs)
            next_value = next_value.flatten()
            self.buffer.compute_returns(next_value, next_dones)

    def _ppo_loss(self, agent, rollout_data):
        _, values, log_prob, entropy = agent(rollout_data.obs, rollout_data.actions)
        values = values.flatten()

        # Normalize advantage
        advantages = rollout_data.advantages
        if self.args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        log_ratio = log_prob - rollout_data.logprobs
        ratio = log_ratio.exp()

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(
            ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
        )
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        if self.args.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.values + th.clamp(
                values - rollout_data.values,
                -self.args.clip_range_vf,
                self.args.clip_range_vf,
            )

        # Value loss using the TD(gae_lambda) target
        value_loss = ((rollout_data.returns - values_pred) ** 2).mean()

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        return policy_loss, value_loss, entropy_loss
