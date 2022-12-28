import numpy as np
import datetime as dt
import gymnasium as gym
import yaml
from types import SimpleNamespace
import logging
import torch as th
from torch.utils.tensorboard import SummaryWriter
from torchmeta.utils.gradient_based import gradient_update_parameters
from copy import deepcopy
from stock_env.envs import *
from stock_env.algos.buffer import RolloutBuffer
from stock_env.algos.agent import MetaAgent
from stock_env.envs import MetaVectorEnv
from stock_env.common.evaluation import evaluate_agent

# GLOBALS
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
global_step = 0


@th.no_grad()
def _meta_collect_rollout(
    args,
    agent,
    buffer,
    envs,
    is_timeout=True,
    params=None,
    writer=None,
):
    device = buffer.device
    buffer.reset()
    obs, _ = envs.reset()
    dones = th.zeros((envs.num_envs,))
    obs = th.Tensor(obs).to(device).to(th.float32)

    for step in range(buffer.num_steps):
        global global_step
        global_step += 1 * envs.num_envs
        actions, values, log_probs, _ = agent.get_action_and_value(obs, params=params)
        values = values.flatten()

        next_obs, rewards, next_terminated, next_truncated, next_infos = envs.step(
            actions.cpu().numpy()
        )
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
                    th.Tensor(final_observation).to(device), params=params
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
                epoch = globals()["epoch"]
                logging.info(
                    f"global_step={global_step}, episodic_return={mean_reward :.2f}, epoch={epoch}"
                )
                if agent.training:
                    writer.add_scalar("metric/train_reward", mean_reward, global_step)
                else:
                    writer.add_scalar("metric/eval_reward", mean_reward, global_step)

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
    args,
    agent,
    buffer,
    params=None,
    writer=None,
):
    pg_losses, vf_losses, ent_losses = [], [], []
    for rollout_data in buffer.get(args.minibatch_size):
        _, values, log_prob, entropy = agent.get_action_and_value(
            rollout_data.obs, rollout_data.actions, params=params
        )
        values = values.flatten()

        advantages = rollout_data.advantages
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        log_ratio = log_prob - rollout_data.logprobs
        ratio = log_ratio.exp()

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        min_policy = th.min(policy_loss_1, policy_loss_2)
        policy_loss = -min_policy.mean()
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
        value_loss = ((rollout_data.returns - values_pred) ** 2).mean()
        vf_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)
        ent_losses.append(entropy_loss.item())

        loss = policy_loss + args.ent_coef * entropy_loss + args.vf_coef * value_loss

    # logging
    if writer is not None:
        if agent.training:
            writer.add_scalar(
                "inner_train/policy_loss", np.mean(pg_losses), global_step
            )
            writer.add_scalar("inner_train/value_loss", np.mean(vf_losses), global_step)
            writer.add_scalar(
                "inner_train/entropy_loss", np.mean(ent_losses), global_step
            )
        else:
            writer.add_scalar("inner_eval/policy_loss", np.mean(pg_losses), global_step)
            writer.add_scalar("inner_eval/value_loss", np.mean(vf_losses), global_step)
            writer.add_scalar(
                "inner_eval/entropy_loss", np.mean(ent_losses), global_step
            )
    return loss


def get_task_loss(args, meta_agent, meta_env, buffer, params=None, writer=None):
    filled_buffer = _meta_collect_rollout(
        args=args,
        agent=meta_agent,
        buffer=buffer,
        envs=meta_env,
        params=params,
        writer=writer,
    )

    loss = _compute_ppo_loss(
        args=args,
        agent=meta_agent,
        buffer=filled_buffer,
        params=params,
        writer=writer,
    )
    return loss


def adapt(args, meta_agent, meta_env, buffer, n_adapt_steps, step_size, writer=None):

    params = None
    l_loss = []
    meta_agent.train()
    meta_env.train(True)
    for _ in range(n_adapt_steps):
        inner_loss = get_task_loss(
            args=args,
            meta_agent=meta_agent,
            meta_env=meta_env,
            buffer=buffer,
            params=params,
            writer=writer,
        )
        l_loss.append(inner_loss.item())  # logging

        meta_agent.zero_grad()
        params = gradient_update_parameters(
            model=meta_agent,
            loss=inner_loss,
            step_size=step_size,
            params=params,
            first_order=(not meta_agent.training),
        )
    return params, np.mean(l_loss)


if __name__ == "__main__":

    env_id = "HalfCheetahDir-v0"

    # read config from yaml
    with open("configs/maml.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = config[env_id]
        args["inner_lr"] = 10 ** (args["inner_lr_hat"])
        args["outer_lr"] = 10 ** (args["outer_lr_hat"])
        args = SimpleNamespace(**args)
        logging.info(args)

    if args.run_name is None:
        run_name = f'maml_{env_id}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        run_name = f'maml_{args.run_name}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    meta_env = MetaVectorEnv([lambda: gym.make(env_id) for _ in range(args.num_envs)])

    eval_meta_env = MetaVectorEnv(
        [lambda: gym.make(env_id) for _ in range(args.num_envs)]
    )
    eval_tasks = eval_meta_env.sample_task(1)
    eval_meta_env.reset_task(eval_tasks[0])

    buffer = RolloutBuffer(
        args.num_steps,
        meta_env,
        device=device,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    meta_agent = MetaAgent(meta_env, hiddens=args.hiddens, activation_fn=th.nn.Tanh).to(
        device
    )
    random_agent = MetaAgent(
        meta_env, hiddens=args.hiddens, activation_fn=th.nn.Tanh
    ).to(device)

    mean, std = evaluate_agent(random_agent, eval_meta_env, args.n_eval_episodes)
    print(f"Mean reward: {mean:.2f} +/- {std: .2f}")

    meta_optimizer = th.optim.Adam(meta_agent.parameters(), lr=args.outer_lr)
    logging.info(meta_agent)
    writer = SummaryWriter(f"log/{run_name}")

    try:
        best_value, save_model = None, False

        for epoch in range(args.epochs):

            tasks = meta_env.sample_task(args.num_tasks)
            outer_loss = th.tensor(0.0, device=device)
            task_inner_losses = []
            for task_idx, task in enumerate(tasks):

                meta_env.reset_task(task)

                meta_agent.train()
                meta_env.train()
                # adapt
                params, inner_loss = adapt(
                    args,
                    meta_agent,
                    meta_env,
                    buffer,
                    step_size=args.inner_lr,
                    n_adapt_steps=1,
                    writer=writer,
                )
                task_inner_losses.append(inner_loss.item())  # logging

                # validation
                meta_env.train(False)
                meta_agent.eval()
                valid_loss = get_task_loss(
                    args, meta_agent, meta_env, buffer, params, writer=writer
                )
                logging.info(
                    f"inner_loss: {inner_loss.item():.3f} valid_loss: {valid_loss.item():.3f}"
                )

                outer_loss += valid_loss

            outer_loss.div_(args.num_tasks)

            meta_optimizer.zero_grad()
            outer_loss.backward()
            # get the gradient norm
            total_norm = th.nn.utils.clip_grad_norm_(
                meta_agent.parameters(), args.max_grad_norm
            )
            meta_optimizer.step()

            # MISC JOBS
            # logging
            writer.add_scalar("outer_train/outer_loss", outer_loss.item(), epoch)
            writer.add_scalar(
                "outer_train/mean_inner_loss_over_tasks",
                np.mean(task_inner_losses),
                epoch,
            )
            writer.add_scalar(
                "outer_train/lr", meta_optimizer.param_groups[0]["lr"], epoch
            )
            writer.add_scalar("outer_train/ent_coef", args.ent_coef, epoch)
            writer.add_scalar("outer_train/total_norm", total_norm.item(), epoch)

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
        # adapt
        meta_agent.train()
        meta_env.train()

        params, inner_loss = adapt(
            args,
            meta_agent,
            eval_meta_env,
            buffer,
            step_size=args.inner_lr,
            n_adapt_steps=1,
        )

        mean, std = evaluate_agent(random_agent, eval_meta_env, args.n_eval_episodes)
        print(f"Mean reward: {mean:.2f} +/- {std: .2f}")

        random_agent.load_state_dict(params)

        mean, std = evaluate_agent(random_agent, eval_meta_env, args.n_eval_episodes)
        print(f"Mean reward: {mean:.2f} +/- {std: .2f}")
        meta_env.close()
        writer.close()

        del meta_agent
        del meta_env
