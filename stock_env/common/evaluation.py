import numpy as np
import torch as th

th.set_default_dtype(th.float32)
import datetime as dt


def evaluate_agent(agent, envs, n_eval_episodes, device="cpu"):

    agent.eval()
    envs.train(False)
    episode_counts = np.zeros(envs.num_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // envs.num_envs for i in range(envs.num_envs)],
        dtype="int",
    )

    next_obs, next_infos = envs.reset()
    next_obs = th.Tensor(next_obs).to(device)

    episode_rewards = []
    episode_lengths = []
    while (episode_counts < episode_count_targets).any():
        with th.no_grad():
            actions = agent.get_action(next_obs, deterministic=True)

        next_obs, rewards, next_terminated, next_truncated, next_infos = envs.step(
            actions.cpu().numpy()
        )
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
    envs.close()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


def play_an_episode(agent, envs, device="cpu"):
    assert envs.num_envs == 1, "Only support Vector Env with single environment"
    envs.train(False)
    agent.eval()
    obs, reset_info = envs.reset()

    print(
        "Ticker: {}, from date {} to date {}".format(
            reset_info["episode_ticker"][0],
            reset_info["from_time"][0],
            reset_info["to_time"][0],
        )
    )
    obs = th.Tensor(obs).to(device)
    dones = np.zeros(envs.num_envs, dtype="bool")

    while not any(dones):
        with th.no_grad():
            actions = agent.get_action(obs, deterministic=True)

        obs, _, terminated, truncated, info = envs.step(actions.cpu().numpy())
        dones = terminated | truncated
        obs = th.Tensor(obs).to(device)

    envs.close()
    return info


def evaluate_sb3_policy(model, envs, n_eval_episodes, device="cpu", deterministic=True):

    episode_counts = np.zeros(envs.num_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // envs.num_envs for i in range(envs.num_envs)],
        dtype="int",
    )

    next_obs, next_infos = envs.reset()
    # next_obs = th.Tensor(next_obs).to(device)

    episode_rewards = []
    episode_lengths = []
    states = None
    episode_starts = np.ones((envs.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        with th.no_grad():
            actions, states = model.predict(
                next_obs,
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )

        next_obs, rewards, next_terminated, next_truncated, next_infos = envs.step(
            actions
        )
        dones = next_terminated | next_truncated

        if any(dones):
            # find which envs are done
            idx = np.where(dones)[0]
            episode_counts[idx] += 1
            final_infos = next_infos["final_info"][idx]

            for info in final_infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
    envs.close()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward
