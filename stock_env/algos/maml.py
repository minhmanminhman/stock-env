import numpy as np
import datetime as dt
import gymnasium as gym

import torch as th
from torch.utils.tensorboard import SummaryWriter
from torchmeta.utils.gradient_based import gradient_update_parameters

from stock_env.envs import *
from stock_env.algos.buffer import RolloutBuffer
from stock_env.algos.agent import MetaAgent
from stock_env.envs import MetaVectorEnv
from stock_env.algos.buffer import RolloutBuffer
from stock_env.common.evaluation import evaluate_agent

@th.no_grad()
def meta_collect_rollout(
    agent, 
    buffer, 
    envs, 
    gamma, 
    device='cpu',
    is_timeout=True,
    params=None,
    writer=None,
):
    epoch = globals()['epoch']
    
    obs, _ = envs.reset()
    dones = th.zeros((envs.num_envs,))
    obs = th.Tensor(obs).to(device).to(th.float32)

    for step in range(buffer.num_steps):
        
        actions, values, log_probs, _ = agent.get_action_and_value(obs, params=params)
        values = values.flatten()

        next_obs, rewards, next_terminated, next_truncated, next_infos = \
            envs.step(actions.cpu().numpy())
        next_obs = th.Tensor(next_obs).to(device)
        rewards = th.Tensor(rewards).to(device).flatten()
        next_dones = th.Tensor(next_terminated | next_truncated).to(device).flatten()
        
        if any(next_dones):
            
            # find which envs are done
            idx = np.where(next_dones)[0]
            if is_timeout:
                final_observation = next_infos["final_observation"][idx][0]
                # calculated value of final observation
                final_value = agent.get_value(
                    th.Tensor(final_observation).to(device),
                    params=params)
                rewards[idx] += gamma * final_value
            
            # logging
            if writer is not None:
                final_infos = next_infos["final_info"][idx]
                for info in final_infos:
                    if "episode" in info.keys():
                        writer.add_scalar("episode/return", info["episode"]["r"], epoch)
                        writer.add_scalar("episode/length", info["episode"]["l"], epoch)
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
    next_value = agent.get_value(next_obs, params=params).flatten()
    buffer.compute_returns(next_value, next_dones)
    return buffer

def ppo_loss(
    agent, 
    buffer, 
    normalize_advantage=True, 
    clip_coef=0.2, 
    clip_range_vf=None,
    ent_coef=0.01,
    vf_coef=0.5,
):
    
    for rollout_data in buffer.get():
        _, values, log_prob, entropy = agent.get_action_and_value(rollout_data.obs, rollout_data.actions)
        values = values.flatten()
        
        advantages = rollout_data.advantages
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        log_ratio = log_prob - rollout_data.logprobs
        ratio = log_ratio.exp()

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        if clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.values + th.clamp(
                values - rollout_data.values, -clip_range_vf, clip_range_vf
            )
        value_loss = ((rollout_data.returns - values_pred) ** 2).mean()

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
    return loss
    
if __name__ == '__main__':
    env_id = 'FAANGTask-v0'
    num_envs = 5
    num_tasks = 5
    num_steps = 25
    epochs = 100
    gamma = 0.99
    seed = 0
    gae_lambda = 0.9
    learning_rate = 1e-3
    inner_lr = 0.4
    is_timeout = True
    n_eval_episodes = 5
    clip_coef = 0.2
    clip_range_vf = 0.2
    vf_coef = 0.19
    ent_coef = 0.02
    max_grad_norm = 5
    normalize_advantage = True
    target_kl = None
    run_name = f'maml_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    task_env = MetaVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)])
    buffer = RolloutBuffer(num_steps, task_env, device=device, gamma=gamma, gae_lambda=gae_lambda)
    agent = MetaAgent(task_env)
    meta_optimizer = th.optim.Adam(agent.parameters(), lr=learning_rate)

    writer = SummaryWriter(f"log/{run_name}")
    try:
        global_step = 0
        inner_losses = []
        for epoch in range(epochs):
            
            agent.zero_grad()
            tasks = task_env.sample_task(num_tasks)
            outer_loss = th.tensor(0., device=device)
            # INNER LOOP
            for task_idx, task in enumerate(tasks):
                global_step += 1 * num_tasks
                task_env.reset_task(task)
                
                agent.train()
                task_env.train()
                buffer = buffer = meta_collect_rollout(
                    agent=agent, 
                    buffer=buffer, 
                    envs=task_env, 
                    gamma=gamma,
                    device=device,
                )
                
                # update agent
                inner_loss = ppo_loss(
                    agent=agent,
                    buffer=buffer,
                    normalize_advantage=normalize_advantage,
                    clip_coef=clip_coef,
                    clip_range_vf=clip_range_vf,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef)
                inner_losses.append(inner_loss.item()) # logging
                
                agent.zero_grad()
                params = gradient_update_parameters(
                    model=agent, 
                    loss=inner_loss, 
                    step_size=inner_lr,)
                
                # validation
                task_env.train(False)
                buffer = meta_collect_rollout(
                    agent=agent, 
                    buffer=buffer, 
                    envs=task_env, 
                    gamma=gamma,
                    device=device,
                    params=params,
                    writer=writer,
                )
                
                valid_loss = ppo_loss(
                    agent=agent,
                    buffer=buffer,
                    normalize_advantage=normalize_advantage,
                    clip_coef=clip_coef,
                    clip_range_vf=clip_range_vf,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,)
                
                outer_loss += valid_loss

            outer_loss.div_(num_tasks)
            
            writer.add_scalar("train/outer_loss", outer_loss.item(), epoch)
            writer.add_scalar("train/mean_inner_loss", np.mean(inner_losses), epoch)
            outer_loss.backward()
            meta_optimizer.step()
    except KeyboardInterrupt:
        pass
    finally:
        eval_tasks = task_env.sample_task(task_env.num_envs)
        for task, env in zip(eval_tasks, task_env.envs):
            env.reset_task(task)
            env.data_loader.train(False)

        mean, std = evaluate_agent(agent, task_env, n_eval_episodes)
        print(f"Mean reward: {mean:.2f} +/- {std: .2f}")
        
        th.save(agent.state_dict(), f"model/{run_name}.pth")
        
        task_env.close()
        writer.close()
        
        del agent
        del task_env