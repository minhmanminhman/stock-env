import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import datetime as dt

from stock_env.algos.agent import Agent
from stock_env.algos.buffer import RolloutBuffer


def make_env(env_id, seed, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == '__main__':
    env_id = 'MountainCarContinuous-v0'
    num_envs = 8
    num_steps = 8
    total_timesteps = 150_000
    epoches = total_timesteps // num_steps // num_envs
    gamma = 0.9999
    gae_lambda = 0.9
    learning_rate = 7.77e-05
    # learning_rate = 1e-04
    seed = 0
    is_timeout = True
    n_eval_episodes = 5
    clip_coef = 0.2
    clip_range_vf = 0.2
    vf_coef = 0.19
    ent_coef = 0.01
    max_grad_norm = 5
    normalize_advantage = True
    target_kl = None
    run_name = f'ppo_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    envs = SyncVectorEnv([make_env(env_id, seed, gamma) for _ in range(num_envs)])
    writer = SummaryWriter(f"log/{run_name}")
    buffer = RolloutBuffer(num_steps, envs, device=device, gamma=gamma, gae_lambda=gae_lambda)

    agent = Agent(envs)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    obs, infos = envs.reset()
    dones = th.zeros((num_envs,))
    obs = th.Tensor(obs).to(device)
    global_step = 0
    n_train_episodes = 0
    
    for epoch in range(epoches):
        """ Rollout to fill the buffer """
        for step in range(num_steps):
            global_step += 1 * num_envs
            with th.no_grad():
                actions, values, log_probs, _ = agent.get_action_and_value(obs)
                values = values.flatten()

            next_obs, rewards, next_terminated, next_truncated, next_infos = \
                envs.step(actions.cpu().numpy())
            next_obs = th.Tensor(next_obs).to(device)
            rewards = th.Tensor(rewards).to(device).flatten()
            next_dones = th.Tensor(next_terminated | next_truncated).to(device).flatten()
            
            if any(next_dones):
                # find which envs are done
                idx = np.where(next_dones)[0]
                n_train_episodes += len(idx)
                # Handle timeout by bootstraping with value function
                # NOTES: for timeout env    
                if is_timeout:
                    final_observation = next_infos["final_observation"][idx][0]
                    # calculated value of final observation
                    with th.no_grad():
                        final_value = agent.get_value(th.Tensor(final_observation).to(device))
                    rewards[idx] += gamma * final_value
                
                # logging
                final_infos = next_infos["final_info"][idx]
                for info in final_infos:
                    if "episode" in info.keys():
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}, n_train_episodes={n_train_episodes}")
                        writer.add_scalar("episode/return", info["episode"]["r"], global_step)
                        writer.add_scalar("episode/length", info["episode"]["l"], global_step)
                        break
            
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
        # Do a complete pass on the rollout buffer
        for rollout_data in buffer.get():
            _, values, log_prob, entropy = agent.get_action_and_value(rollout_data.obs, rollout_data.actions)
            values = values.view(-1)
            
            # Normalize advantage
            advantages = rollout_data.advantages
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            log_ratio = log_prob - rollout_data.logprobs
            ratio = log_ratio.exp()

            with th.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses.append(policy_loss.item())

            if clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip the difference between old and new value
                # NOTE: this depends on the reward scaling
                values_pred = rollout_data.values + th.clamp(
                    values - rollout_data.values, -clip_range_vf, clip_range_vf
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

            loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

            # Calculate approximate form of reverse KL Divergence for early stopping
            # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
            # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
            # and Schulman blog: http://joschu.net/blog/kl-approx.html
            with th.no_grad():
                log_ratio = log_prob - rollout_data.logprobs
                approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        y_pred, y_true = rollout_data.values.cpu().numpy(), rollout_data.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logs
        writer.add_scalar("train/entropy_loss", np.mean(entropy_losses), global_step)
        writer.add_scalar("train/policy_gradient_loss", np.mean(pg_losses), global_step)
        writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
        writer.add_scalar("train/approx_kl", np.mean(approx_kl_divs), global_step)
        writer.add_scalar("train/clip_fraction", np.mean(clipfracs), global_step)
        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/explained_variance", explained_var, global_step)
    
    envs.close()
    writer.close()

    
    # evaluation
    episode_rewards = []
    episode_lengths = []
    episode_counts = np.zeros(num_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // num_envs for i in range(num_envs)], dtype="int")
    
    next_obs, next_infos = envs.reset()
    next_obs = th.Tensor(next_obs).to(device)
    
    while (episode_counts < episode_count_targets).any():
        with th.no_grad():
            actions = agent.get_action(next_obs)
        
        next_obs, rewards, next_terminated, next_truncated, next_infos = \
            envs.step(actions.cpu().numpy())
        dones = next_terminated | next_truncated
        next_obs = th.Tensor(next_obs).to(device)
        
        if any(dones):
            # find which envs are done
            idx = np.where(dones)[0]
            episode_counts[idx] += 1
            final_infos = next_infos["final_info"][idx]
            for info in final_infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward: .2f}")

    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    print(f"Mean lengths: {mean_length:.2f} +/- {std_length: .2f}")
    
    th.save(agent.state_dict(), "model/mountain_car_agent.pth")
    
    envs.close()