import numpy as np
import datetime as dt
import gymnasium as gym
import yaml
from types import SimpleNamespace
import torch as th
from torch.utils.tensorboard import SummaryWriter
from torchmeta.utils.gradient_based import gradient_update_parameters

from stock_env.envs import *
from stock_env.algos.buffer import RolloutBuffer
from stock_env.algos.agent import MetaAgent
from stock_env.envs import MetaVectorEnv
from stock_env.common.evaluation import evaluate_agent

@th.no_grad()
def _meta_collect_rollout(
    agent, 
    buffer, 
    envs, 
    gamma, 
    is_timeout=True,
    params=None,
    writer=None,
):  
    device = buffer.device
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
                        
                        epoch = globals()['epoch']
                        global_step = globals()['global_step']
                        
                        if global_step % 10 == 0:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r'].item() :.2f}, epoch={epoch}")
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

def _compute_ppo_loss(
    agent, 
    buffer, 
    normalize_advantage=True, 
    clip_coef=0.2, 
    clip_range_vf=None,
    ent_coef=0.01,
    vf_coef=0.5,
    params=None,
):
    
    for rollout_data in buffer.get():
        _, values, log_prob, entropy = agent.get_action_and_value(rollout_data.obs, rollout_data.actions, params=params)
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

def get_task_loss(
    args,
    meta_agent, 
    meta_env, 
    buffer, 
    params=None, 
    writer=None
):
    filled_buffer = _meta_collect_rollout(
        agent=meta_agent, 
        buffer=buffer, 
        envs=meta_env, 
        gamma=args.gamma,
        params=params,
        writer=writer,)
    
    loss = _compute_ppo_loss(
        agent=meta_agent,
        buffer=filled_buffer,
        params=params,
        normalize_advantage=args.normalize_advantage,
        clip_coef=args.clip_coef,
        clip_range_vf=args.clip_range_vf,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,)
    return loss

def adapt(args, meta_agent, meta_env, buffer, n_adapt_steps):
    
    params = None
    l_loss = []
    for _ in range(n_adapt_steps):
        inner_loss = get_task_loss(
            args=args,
            meta_agent=meta_agent,
            meta_env=meta_env,
            buffer=buffer,
            params=params,
        )
        l_loss.append(inner_loss.item()) # logging
        
        meta_agent.zero_grad()
        params = gradient_update_parameters(
            model=meta_agent, 
            loss=inner_loss, 
            step_size=args.inner_lr,
            params=params,
            first_order=(not meta_agent.training)
        )
    return params, np.mean(l_loss)

if __name__ == '__main__':
    
    env_id = 'MiniFAANG-v0'
    
    # read config from yaml
    with open('configs/maml.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = config[env_id]
        args = SimpleNamespace(**args)

    if args.run_name is None:
        run_name = f'maml_minifaang_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        run_name = f'{args.run_name}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    meta_env = MetaVectorEnv([lambda: gym.make(env_id) for _ in range(args.num_envs)])
    buffer = RolloutBuffer(args.num_steps, meta_env, device=device, gamma=args.gamma, gae_lambda=args.gae_lambda)
    meta_agent = MetaAgent(meta_env).to(device)
    meta_optimizer = th.optim.Adam(meta_agent.parameters(), lr=args.outer_lr)

    writer = SummaryWriter(f"log/{run_name}")
    try:
        global_step, best_value, save_model = 0, None, False
        
        for epoch in range(args.epochs):
            
            tasks = meta_env.sample_task(args.num_tasks)
            outer_loss = th.tensor(0., device=device)
            task_inner_losses = []
            for task_idx, task in enumerate(tasks):
                global_step += 1 * args.num_tasks
                meta_env.reset_task(task)
                
                meta_agent.train()
                meta_env.train()
                # adapt
                params, inner_loss = adapt(args, meta_agent, meta_env, buffer, n_adapt_steps=1)
                task_inner_losses.append(inner_loss.item()) # logging
                
                # validation
                meta_env.train(False)
                valid_loss = get_task_loss(args, meta_agent, meta_env, buffer, params)
                
                outer_loss += valid_loss

            outer_loss.div_(args.num_tasks)
            
            meta_optimizer.zero_grad()
            outer_loss.backward()
            meta_optimizer.step()
            
            # MISC JOBS
            # logging
            writer.add_scalar("train/outer_loss", outer_loss.item(), epoch)
            writer.add_scalar("train/mean_inner_loss_over_tasks", np.mean(task_inner_losses), epoch)
            # Save best model
            if (best_value is None) or (best_value > outer_loss.item()):
                best_value = outer_loss.item()
                save_model = True
            else:
                save_model = False

            if save_model:
                th.save(meta_agent.state_dict(), f"model/{run_name}.pth")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        eval_tasks = meta_env.sample_task(meta_env.num_envs)
        for task, env in zip(eval_tasks, meta_env.envs):
            env.reset_task(task)
            env.data_loader.train(False)
        
        meta_agent.load_state_dict(th.load(f"model/{run_name}.pth"))
        meta_agent.eval()
        mean, std = evaluate_agent(meta_agent, meta_env, args.n_eval_episodes)
        print(f"Mean reward: {mean:.2f} +/- {std: .2f}")
        
        meta_env.close()
        writer.close()
        
        del meta_agent
        del meta_env