import numpy as np
import torch as th
th.set_default_dtype(th.float32)

def evaluate_agent(agent, envs, n_eval_episodes, device='cpu'):
    
    episode_counts = np.zeros(envs.num_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // envs.num_envs for i in range(envs.num_envs)], dtype="int")
    
    next_obs, next_infos = envs.reset()
    next_obs = th.Tensor(next_obs).to(device)
    
    episode_rewards = []
    episode_lengths = []
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
    envs.close()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

@th.no_grad()
def play_an_episode(agent, envs, device='cpu'):
    
    obs, _ = envs.reset()
    obs = th.Tensor(obs).to(device)
    dones = np.zeros(envs.num_envs, dtype="bool")
    
    while not any(dones):
        actions = agent.get_action(obs)
        
        obs, _, terminated, truncated, _ = envs.step(actions.cpu().numpy())
        dones = terminated | truncated
        obs = th.Tensor(obs).to(device)
        
    envs.close()
    return envs