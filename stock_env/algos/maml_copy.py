import numpy as np
import datetime as dt
import gymnasium as gym
import yaml
from types import SimpleNamespace
import logging
import torch as th
from torch.utils.tensorboard import SummaryWriter
from torchmeta.utils.gradient_based import gradient_update_parameters
import higher
from stock_env.envs import *
from stock_env.algos.buffer import RolloutBuffer
from stock_env.algos.agent import Agent
from stock_env.envs import MetaVectorStockEnv
from stock_env.common.evaluation import evaluate_agent
from stock_env.common.common_utils import get_linear_fn, open_config

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class ModelAgnosticMetaLearning:
    def __init__(self, env_id, test_env_id):

        self.env_id = env_id
        self.test_env_id = test_env_id
        self.args = open_config("configs/maml.yaml", env_id=env_id)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.envs = self._make_envs(self.env_id)
        self.buffer = RolloutBuffer(
            num_steps=self.args.num_steps,
            envs=self.envs,
            device=self.device,
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
        )
        self.agent = self._make_agent()
        self.adapted_agent = self._make_agent()

        # optimizer
        self.inner_optimiser = th.optim.SGD(
            self.agent.parameters(), lr=self.args.inner_lr, momentum=0.9
        )
        self.meta_optimizer = th.optim.Adam(
            self.agent.parameters(), lr=self.args.outer_lr
        )
        # save path and logging
        if self.args.run_name is None:
            self.run_name = (
                f'maml_{env_id}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
        else:
            self.run_name = f'maml_{self.args.run_name}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.log_path = f"log/{self.run_name}"
        self.model_path = f"model/{self.run_name}.pth"
        self.writer = SummaryWriter(self.log_path)

        # train utils
        self.logger = logging.getLogger(__name__)
        self.best_value, self.save_model = None, False
        self.train_step, self.test_step = 0, 0

    def _make_envs(self, env_id):
        return MetaVectorStockEnv(
            [lambda: gym.make(env_id) for _ in range(self.args.num_envs)]
        )

    def _make_agent(self):
        return Agent(self.envs, self.args.hiddens).to(self.device)

    @th.no_grad()
    def _collect_rollout(self, agent):

        self.buffer.reset()
        obs, _ = self.envs.reset()
        dones = th.zeros((self.envs.num_envs,))
        obs = th.Tensor(obs).to(self.device).to(th.float32)
        update_step = 0

        for step in range(self.buffer.num_steps):
            update_step += 1 * self.envs.num_envs
            actions, values, log_probs, _ = agent.get_action_and_value(obs)
            values = values.flatten()

            (
                next_obs,
                rewards,
                next_terminated,
                next_truncated,
                next_infos,
            ) = self.envs.step(actions.cpu().numpy())

            next_obs = th.Tensor(next_obs).to(self.device)
            rewards = th.Tensor(rewards).to(self.device).flatten()
            next_dones = np.logical_or(next_terminated, next_truncated)
            next_dones = th.Tensor(next_dones).to(self.device).flatten()

            if next_dones.any():

                # find which envs are done
                idx = np.where(next_dones)[0]
                if self.args.is_timeout:
                    final_observation = next_infos["final_observation"][idx][0]
                    # calculated value of final observation
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
                if self.agent.training:
                    self.train_step += update_step
                    self.writer.add_scalar(
                        "metric/train_reward", mean_reward, self.train_step
                    )
                    self.logger.info(
                        f"global_step={self.train_step}, episodic_return={mean_reward :.2f}, epoch={self.epoch}"
                    )
                else:
                    self.test_step += update_step
                    self.writer.add_scalar(
                        "metric/eval_reward", mean_reward, self.test_step
                    )
                    self.logger.info(
                        f"global_step={self.test_step}, episodic_return={mean_reward :.2f}, epoch={self.epoch}"
                    )

            # add to buffer
            self.buffer.add(
                index=step,
                obs=obs,
                actions=actions,
                logprobs=log_probs,
                rewards=rewards,
                dones=dones,
                values=values,
            )
            obs = next_obs
            dones = next_dones

        # compute returns
        next_value = agent.get_value(next_obs).flatten()
        self.buffer.compute_returns(next_value, next_dones)

    def _ppo_loss(self, agent, rollout_data):
        _, values, log_prob, entropy = agent.get_action_and_value(
            rollout_data.obs, rollout_data.actions
        )
        values = values.flatten()

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
        min_policy = th.min(policy_loss_1, policy_loss_2)
        policy_loss = -min_policy.mean()

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
        value_loss = ((rollout_data.returns - values_pred) ** 2).mean()

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)
        loss = (
            policy_loss
            + self.args.ent_coef * entropy_loss
            + self.args.vf_coef * value_loss
        )
        return loss

    def train(self):
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            tasks = self.envs.sample_task(self.args.num_tasks)
            outer_loss = th.tensor(0.0, device=self.device)

            # inner loop
            self.meta_optimizer.zero_grad()
            train_task_loss = []
            eval_task_loss = []
            for task in tasks:

                self.envs.reset_task(task)
                self.agent.train()
                self.envs.train()
                with higher.innerloop_ctx(
                    self.agent, self.inner_optimiser, copy_initial_weights=False
                ) as (fmodel, diffopt):

                    self._collect_rollout(fmodel)
                    train_batch_loss = []
                    for rollout_data in self.buffer.get(self.args.minibatch_size):
                        inner_loss = self._ppo_loss(fmodel, rollout_data)
                        th.nn.utils.clip_grad_norm_(
                            fmodel.parameters(), self.args.max_grad_norm
                        )
                        diffopt.step(inner_loss)

                        train_batch_loss.append(inner_loss.item())
                    train_task_loss.append(np.mean(train_batch_loss))

                    # validate adaption
                    self._collect_rollout(fmodel)
                    for rollout_data in self.buffer.get():
                        outer_loss += self._ppo_loss(fmodel, rollout_data)

            # get the gradient norm
            outer_loss.div_(len(tasks))
            outer_loss.backward()
            total_norm = th.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.args.max_grad_norm
            )
            self.meta_optimizer.step()

            # MISC JOBS
            # logging
            self.writer.add_scalar("outer_train/outer_loss", outer_loss.item(), epoch)
            self.writer.add_scalar(
                "outer_train/inner_loss",
                np.mean(train_task_loss),
                epoch,
            )
            # self.writer.add_scalar(
            #     "outer_train/lr", self.meta_optimizer.param_groups[0]["lr"], epoch
            # )
            # self.writer.add_scalar("outer_train/ent_coef", args.ent_coef, epoch)
            self.writer.add_scalar("outer_train/total_norm", total_norm.item(), epoch)

            # Save best model
            if (self.best_value is None) or (self.best_value > np.mean(eval_task_loss)):
                self.best_value = np.mean(eval_task_loss)
                self.save_model = True
            else:
                self.save_model = False

            if self.save_model:
                th.save(self.agent.state_dict(), self.model_path)

    def close(self):
        self.envs.close()
        self.writer.close()
        del self.agent
        del self.adapted_agent
        del self.envs

    def test(self, eval_tasks, model_path=None):
        eval_meta_env = self._make_envs(self.test_env_id)
        random_agent = self._make_agent()
        random_agent.load_state_dict(th.load(model_path))
        for task in eval_tasks:

            self.adapted_agent.load_state_dict(th.load(model_path))
            optimizer = th.optim.SGD(
                self.adapted_agent.parameters(), lr=self.args.inner_lr
            )

            eval_meta_env.reset_task(task)
            self.adapted_agent.train()
            eval_meta_env.train()

            # adapt
            self._collect_rollout(self.adapted_agent)
            for rollout_data in self.buffer.get(self.args.minibatch_size):
                inner_loss = self._ppo_loss(self.adapted_agent, rollout_data)
                optimizer.zero_grad()
                inner_loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.adapted_agent.parameters(), self.args.max_grad_norm
                )
                optimizer.step()

            self.adapted_agent.eval()
            eval_meta_env.train(False)

            mean, std = evaluate_agent(
                random_agent, eval_meta_env, self.args.n_eval_episodes
            )
            print(
                f"Task {task} | random_agent | Mean reward: {mean:.2f} +/- {std: .2f}"
            )

            mean, std = evaluate_agent(
                self.adapted_agent, eval_meta_env, self.args.n_eval_episodes
            )
            print(
                f"Task {task} | adapted_agent | Mean reward: {mean:.2f} +/- {std: .2f}"
            )

            del optimizer


if __name__ == "__main__":
    try:
        algo = ModelAgnosticMetaLearning(
            env_id="SP500-v0",
            test_env_id="VNALL-v0",
        )
        algo.train()
    except KeyboardInterrupt:
        pass
    finally:
        tasks = ["SSI", "HPG", "VND", "HAH", "VHC"]
        model_path = algo.model_path
        algo.test(eval_tasks=tasks, model_path=model_path)
        algo.close()
