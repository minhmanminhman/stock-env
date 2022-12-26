import gymnasium as gym
import yaml
from types import SimpleNamespace
import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import datetime as dt
from gymnasium.vector import SyncVectorEnv
from stock_env.envs import *
from stock_env.algos.agent import Agent
from stock_env.algos.buffer import RolloutBuffer
from stock_env.common.evaluation import evaluate_agent


def make_env(name, gamma=0.99):
    def _thunk():
        env = gym.make(name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return _thunk


if __name__ == "__main__":
    env_id = "MountainCarContinuous-v0"

    # read config from yaml
    with open("configs/ppo.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = config[env_id]
        args = SimpleNamespace(**args)

    # calculated params
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    epochs = args.total_timesteps // batch_size

    if args.run_name is None:
        run_name = f'ppo_{env_id}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        run_name = f'ppo_{args.run_name}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    envs = SyncVectorEnv([make_env(env_id, args.gamma) for _ in range(args.num_envs)])
    eval_envs = SyncVectorEnv(
        [make_env(env_id, args.gamma) for _ in range(args.num_envs)]
    )
    writer = SummaryWriter(f"log/{run_name}")
    buffer = RolloutBuffer(
        args.num_steps,
        envs,
        device=device,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    agent = Agent(envs, hiddens=args.hiddens)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs, infos = envs.reset()
    dones = th.zeros((args.num_envs,))
    obs = th.Tensor(obs).to(device)
    global_step = 0
    n_train_episodes = 0
    best_value = None
    buffer.reset()

    try:
        for epoch in range(epochs):

            agent.train()
            """Rollout to fill the buffer"""
            for step in range(args.num_steps):
                global_step += 1 * args.num_envs
                with th.no_grad():
                    actions, values, log_probs, _ = agent.get_action_and_value(obs)
                    values = values.flatten()

                (
                    next_obs,
                    rewards,
                    next_terminated,
                    next_truncated,
                    next_infos,
                ) = envs.step(actions.cpu().numpy())
                next_obs = th.Tensor(next_obs).to(device)
                rewards = th.Tensor(rewards).to(device).flatten()
                next_dones = (
                    th.Tensor(next_terminated | next_truncated).to(device).flatten()
                )

                if any(next_dones):
                    # find which envs are done
                    idx = np.where(next_dones)[0]
                    n_train_episodes += len(idx)
                    # Handle timeout by bootstraping with value function
                    # NOTES: for timeout env
                    if args.is_timeout:
                        final_observation = next_infos["final_observation"][idx][0]
                        # calculated value of final observation
                        with th.no_grad():
                            final_value = agent.get_value(
                                th.Tensor(final_observation).to(device)
                            )
                        rewards[idx] += args.gamma * final_value

                    # logging
                    if writer is not None:
                        final_infos = next_infos["final_info"][idx]
                        mean_reward = np.mean(
                            [
                                info["episode"]["r"]
                                for info in final_infos
                                if "episode" in info.keys()
                            ]
                        )
                        print(
                            f"global_step={global_step}, episodic_return={mean_reward :.2f}, epoch={epoch}"
                        )
                        writer.add_scalar(
                            "metric/train_reward", mean_reward, global_step
                        )

                # add to buffer
                buffer.add(
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
            with th.no_grad():
                next_value = agent.get_value(next_obs)
                next_value = next_value.flatten()
                buffer.compute_returns(next_value, next_dones)

            """ PPO update """
            approx_kl_divs, pg_losses, clipfracs = [], [], []
            value_losses, entropy_losses = [], []
            for _ in range(args.gradient_steps):
                # Do a complete pass on the rollout buffer
                for rollout_data in buffer.get(minibatch_size):
                    _, values, log_prob, entropy = agent.get_action_and_value(
                        rollout_data.obs, rollout_data.actions
                    )
                    values = values.view(-1)

                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if args.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                    # ratio between old and new policy, should be one at the first iteration
                    log_ratio = log_prob - rollout_data.logprobs
                    ratio = log_ratio.exp()

                    with th.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())

                    if args.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.values + th.clamp(
                            values - rollout_data.values,
                            -args.clip_range_vf,
                            args.clip_range_vf,
                        )

                    # Value loss using the TD(gae_lambda) target
                    # print(rollout_data.returns)
                    # print(values_pred)
                    value_loss = ((rollout_data.returns - values_pred) ** 2).mean()
                    # print(value_loss)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = (
                        policy_loss
                        + args.ent_coef * entropy_loss
                        + args.vf_coef * value_loss
                    )

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.logprobs
                        approx_kl_div = (
                            th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        )
                        approx_kl_divs.append(approx_kl_div)

                    if args.target_kl is not None:
                        if approx_kl > args.target_kl:
                            break

                    # Optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    total_norm = th.nn.utils.clip_grad_norm_(
                        agent.parameters(), args.max_grad_norm
                    )
                    optimizer.step()

            y_pred, y_true = (
                rollout_data.values.cpu().numpy(),
                rollout_data.returns.cpu().numpy(),
            )
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # Logs
            writer.add_scalar(
                "train/entropy_loss", np.mean(entropy_losses), global_step
            )
            writer.add_scalar(
                "train/policy_gradient_loss", np.mean(pg_losses), global_step
            )
            writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
            writer.add_scalar("train/approx_kl", np.mean(approx_kl_divs), global_step)
            writer.add_scalar("train/clip_fraction", np.mean(clipfracs), global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/explained_variance", explained_var, global_step)
            writer.add_scalar("train/total_norm", total_norm.item(), global_step)

            # Save best model
            mean, std = evaluate_agent(agent, eval_envs, args.n_eval_episodes)
            print(f"Mean reward: {mean:.2f} +/- {std: .2f}")
            writer.add_scalar("metric/test_reward", mean, epoch)

            if (best_value is None) or (best_value < mean):
                best_value = mean
                save_model = True
            else:
                save_model = False

            if save_model:
                th.save(agent.state_dict(), f"model/{run_name}.pth")

    except KeyboardInterrupt:
        pass

    finally:
        mean, std = evaluate_agent(agent, eval_envs, args.n_eval_episodes)
        print(f"Mean reward: {mean:.2f} +/- {std: .2f}")
        envs.close()
        writer.close()

        del agent
        del envs
