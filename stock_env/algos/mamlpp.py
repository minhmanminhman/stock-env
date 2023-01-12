import numpy as np
import datetime as dt
from typing import List
import logging
from copy import deepcopy
import torch as th
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import higher
from stock_env.algos.base_ppo import BasePPO
from stock_env.algos.callback import MAMLLogCallback
from stock_env.common.evaluation import evaluate_agent
from stock_env.algos.agent import Agent


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    th.manual_seed(seed=torch_seed)

    return rng


class MAMLPlusPlusAgent(nn.Module):
    def __init__(
        self,
        args,
        envs,
        activation_fn,
    ):
        super().__init__()
        self.args = args
        self.rng = set_torch_seed(seed=args.seed)
        self.agent = Agent(envs, args.hiddens, activation_fn)

        param_groups = [
            {"params": p, "lr": self.args.inner_optim_kwargs["lr"]}
            for p in self.agent.parameters()
        ]
        # print(len(param_groups))
        self.inner_opt = th.optim.SGD(param_groups, **self.args.inner_optim_kwargs)

        t = higher.optim.get_trainable_opt_params(self.inner_opt)
        self._lrs = nn.ParameterList(map(nn.Parameter, t["lr"]))
        # lrs = [th.ones(1) * p["lr"] for p in param_groups]
        # self.lrs = nn.ParameterList([nn.Parameter(lr) for lr in lrs])
        # self.lrs = t["lr"]

        # print("Outer Loop parameters")
        # param_shapes = []
        # for name, param in self.named_parameters():
        #     print(name, param.shape, param.device, param.requires_grad)
        #     param_shapes.append(param.shape)
        # print(f"n_params: {sum(map(np.prod, param_shapes))}")

    @property
    def lrs(self):
        for lr in self._lrs:
            lr.data[lr < 1e-4] = 1e-4
        return self._lrs

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param


class MAMLPlusPlus(BasePPO):
    def __init__(self, args, envs, activation_fn: nn.Module = nn.LeakyReLU):
        super().__init__(args, envs, agent=None)

        self.meta_agent = MAMLPlusPlusAgent(args, envs, activation_fn=activation_fn)
        self.agent = self.meta_agent.agent
        self.untrained_agent = deepcopy(self.agent)
        self.lrs = [p.data for p in self.meta_agent.lrs]

        # optimizer
        self.meta_optimizer = th.optim.Adam(
            self.meta_agent.trainable_parameters(), lr=self.args.meta_lr
        )
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.meta_optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.min_lr,
        )
        # save path and logging
        if self.args.run_name is None:
            env_id = envs.envs[0].spec.id
            self.run_name = (
                f'mamlpp_{env_id}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
        else:
            self.run_name = f'mamlpp_{self.args.run_name}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.log_path = f"log/{self.run_name}"
        self.model_path = f"model/{self.run_name}.pth"
        self.writer = SummaryWriter(self.log_path)
        self.logger = logging.getLogger(__name__)
        self.callback = MAMLLogCallback()
        self.callback.init_callback(self.logger, self.writer)

        # train utils
        self.best_value, self.save_model = None, False
        self.train_step, self.test_step = 0, 0

    def close(self):
        self.envs.close()
        self.writer.close()
        del self

    def _importance_vector(self):
        length = self.args.inner_steps
        init_value = 1 / length
        min_value = 0.03 / length
        decay = 1 / self.args.msl_epoch / length
        init_weights = np.ones(shape=(length)) * init_value

        weights = init_weights.copy()
        weights[:-1] -= self.epoch * decay
        weights[:-1] = np.maximum(weights[:-1], min_value)
        weights[-1] = 1 - np.sum(weights[:-1])
        # self.logger.info(f"importance vector: {weights}")
        weights = th.Tensor(weights).to(self.device)
        return weights

    def _train(self):
        """inner loop train"""

        with higher.innerloop_ctx(
            self.meta_agent.agent, self.meta_agent.inner_opt, copy_initial_weights=False
        ) as (fmodel, diffopt):
            outer_losses = []
            importance_vector = self._importance_vector()
            for step in range(self.args.inner_steps):
                self.envs.train()
                fmodel.train()
                self._env_reset()
                self._collect_rollout(agent=fmodel, callback=self.callback)

                # adaption
                train_batch_loss = []
                for rollout_data in self.buffer.get(self.args.minibatch_size):
                    policy_loss, value_loss, entropy_loss = self._ppo_loss(
                        agent=fmodel, rollout_data=rollout_data
                    )
                    inner_loss = (
                        policy_loss
                        + self.args.ent_coef * entropy_loss
                        + self.args.vf_coef * value_loss
                    )
                    diffopt.step(inner_loss, override={"lr": self.meta_agent.lrs})

                    train_batch_loss.append(inner_loss.item())
                mean_train_task_loss = np.mean(train_batch_loss)

                # validate adaption
                self.envs.train(False)
                fmodel.eval()
                self._env_reset()
                self._collect_rollout(agent=fmodel, callback=self.callback)

                for rollout_data in self.buffer.get():
                    policy_loss, value_loss, entropy_loss = self._ppo_loss(
                        fmodel, rollout_data
                    )
                    _outer_loss = (
                        policy_loss
                        + self.args.ent_coef * entropy_loss
                        + self.args.vf_coef * value_loss
                    )
                if self.epoch < self.args.msl_epoch:
                    outer_losses.append(_outer_loss * importance_vector[step])
                else:
                    if step == self.args.inner_steps - 1:
                        outer_losses.append(_outer_loss)

            outer_loss = th.sum(th.stack(outer_losses))
            self.callback.update_locals(locals(), globals())
            self.callback.on_inner_train()
        return outer_loss

    def learn(self):

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            tasks = self.envs.sample_task(self.args.num_tasks)
            outer_loss = th.tensor(0.0, device=self.device)

            # inner loop
            self.meta_agent.zero_grad()
            for task in tasks:
                self.envs.reset_task(task)
                outer_loss += self._train()

            # get the gradient norm
            outer_loss.div_(len(tasks))
            self.meta_optimizer.zero_grad()
            outer_loss.backward()
            total_norm = th.nn.utils.clip_grad_norm_(
                self.meta_agent.trainable_parameters(), self.args.max_grad_norm
            )
            self.meta_optimizer.step()
            self.scheduler.step(epoch=self.epoch)

            self.callback.update_locals(locals(), globals())
            self.callback.on_outer_train()

            # Save best model
            if (self.best_value is None) or (self.best_value > outer_loss.item()):
                self.best_value = outer_loss.item()
                self.save_model = True
            else:
                self.save_model = False

            if self.save_model:
                th.save(self.agent.state_dict(), self.model_path)
                self.logger.info(f"Save best model at epoch={epoch}")

    def test(self, test_args, test_envs, tasks=None):

        if tasks is None:
            tasks = test_envs.sample_task(test_args.num_tasks)
        untrained_agent = deepcopy(self.untrained_agent)
        adapt_agent = deepcopy(self.untrained_agent)
        means = []
        untrained_means = []
        for task in tasks:
            test_envs.reset_task(task)

            adapt_agent.load_state_dict(th.load(self.model_path))
            optimizer = th.optim.SGD(
                adapt_agent.parameters(), **self.args.inner_optim_kwargs
            )

            adapt_agent.train()
            test_envs.train()
            self._env_reset(envs=test_envs)

            # adapt
            self._collect_rollout(adapt_agent, callback=None, envs=test_envs)
            for rollout_data in self.buffer.get(test_args.minibatch_size):
                policy_loss, value_loss, entropy_loss = self._ppo_loss(
                    agent=adapt_agent, rollout_data=rollout_data
                )
                inner_loss = (
                    policy_loss
                    + test_args.ent_coef * entropy_loss
                    + test_args.vf_coef * value_loss
                )
                optimizer.zero_grad()
                inner_loss.backward()
                th.nn.utils.clip_grad_norm_(
                    adapt_agent.parameters(), test_args.max_grad_norm
                )
                optimizer.step()

            adapt_agent.eval()
            test_envs.train(False)

            # evaluate untrained agent
            mean, std = evaluate_agent(
                untrained_agent, test_envs, test_args.n_eval_episodes
            )
            untrained_means.append(mean)
            self.logger.info(
                f"Task {task} | untrained_agent | Mean reward: {mean:.2f} +/- {std: .2f}"
            )

            # evaluate adapted agent
            mean, std = evaluate_agent(
                adapt_agent, test_envs, test_args.n_eval_episodes
            )
            means.append(mean)
            self.logger.info(
                f"Task {task} | adapted_agent | Mean reward: {mean:.2f} +/- {std: .2f}"
            )
        self.logger.info(
            f"Test | adapted_agent | Mean reward across tasks: {np.mean(means):.2f} +/- {np.std(means): .2f}"
        )
        self.logger.info(
            f"Test | untrained_agent | Mean reward across tasks: {np.mean(untrained_means):.2f} +/- {np.std(untrained_means): .2f}"
        )
