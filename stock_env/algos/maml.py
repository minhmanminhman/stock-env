import numpy as np
import datetime as dt
import logging
from copy import deepcopy
import torch as th
from torch.utils.tensorboard import SummaryWriter
import higher

from stock_env.algos.base_ppo import BasePPO
from stock_env.algos.callback import MAMLLogCallback
from stock_env.common.evaluation import evaluate_agent


class MAML(BasePPO):
    def __init__(self, args, envs, agent):
        super().__init__(args, envs, agent)

        self.untrained_agent = deepcopy(self.agent)
        self.train_envs = self.envs

        # optimizer
        self.inner_optimiser = th.optim.SGD(
            self.agent.parameters(), lr=self.args.inner_lr, momentum=0.9
        )
        self.meta_optimizer = th.optim.Adam(
            self.agent.parameters(), lr=self.args.outer_lr
        )
        # save path and logging
        if self.args.run_name is None:
            env_id = envs.envs[0].spec.id
            self.run_name = (
                f'maml_{env_id}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
        else:
            self.run_name = f'maml_{self.args.run_name}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
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

    def _train(self):
        """inner loop train"""

        with higher.innerloop_ctx(
            self.agent, self.inner_optimiser, copy_initial_weights=False
        ) as (fmodel, diffopt):

            self.envs.train()
            fmodel.train()
            self._env_reset()
            self._collect_rollout(agent=fmodel, callback=self.callback)

            # adaption
            train_batch_loss, total_inner_norm = [], []
            for rollout_data in self.buffer.get(self.args.minibatch_size):
                policy_loss, value_loss, entropy_loss = self._ppo_loss(
                    agent=fmodel, rollout_data=rollout_data
                )
                inner_loss = (
                    policy_loss
                    + self.args.ent_coef * entropy_loss
                    + self.args.vf_coef * value_loss
                )
                _inner_total_norm = th.nn.utils.clip_grad_norm_(
                    fmodel.parameters(), self.args.max_grad_norm
                )
                diffopt.step(inner_loss)

                train_batch_loss.append(inner_loss.item())
                total_inner_norm.append(_inner_total_norm.item())
            mean_train_task_loss = np.mean(train_batch_loss)
            mean_inner_norm = np.mean(total_inner_norm)

            # validate adaption
            self.envs.train(False)
            fmodel.eval()
            self._env_reset()
            self._collect_rollout(agent=fmodel, callback=self.callback)
            for rollout_data in self.buffer.get():
                policy_loss, value_loss, entropy_loss = self._ppo_loss(
                    fmodel, rollout_data
                )
                outer_loss = (
                    policy_loss
                    + self.args.ent_coef * entropy_loss
                    + self.args.vf_coef * value_loss
                )
                outer_loss.backward()
            self.callback.update_locals(locals(), globals())
            self.callback.on_inner_train()
        return outer_loss

    def learn(self):

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            tasks = self.envs.sample_task(self.args.num_tasks)
            outer_loss = th.tensor(0.0, device=self.device)

            # inner loop
            self.meta_optimizer.zero_grad()
            for task in tasks:
                self.envs.reset_task(task)
                outer_loss += self._train()

            # get the gradient norm
            outer_loss.div_(len(tasks))
            # outer_loss.backward()
            total_norm = th.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.args.max_grad_norm
            )
            self.meta_optimizer.step()

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
                adapt_agent.parameters(), lr=test_args.inner_lr, momentum=0.9
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
