import torch as th
import torch.optim as optim

from stock_env.algos.base_ppo import BasePPO
from stock_env.algos.utils import explained_variance
from stock_env.algos.callback import PPOLogCallback
from stock_env.common.evaluation import evaluate_agent


class PPO(BasePPO):
    def __init__(self, args, envs, agent):
        super().__init__(args, envs, agent)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate)
        self.callback = PPOLogCallback()
        self.callback.init_callback(self.logger, self.writer)

        # save best model
        self.best_value = None
        self.save_model = True

    def _train(self):

        pg_losses, value_losses, entropy_losses = [], [], []
        for _ in range(self.args.gradient_steps):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.buffer.get(self.args.minibatch_size):
                policy_loss, value_loss, entropy_loss = self._ppo_loss(
                    agent=self.agent, rollout_data=rollout_data
                )
                loss = (
                    policy_loss
                    + self.args.ent_coef * entropy_loss
                    + self.args.vf_coef * value_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                total_norm = th.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.args.max_grad_norm
                )
                self.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        explained_var = explained_variance(
            y_pred=self.buffer.values.flatten().cpu().numpy(),
            y_true=self.buffer.returns.flatten().cpu().numpy(),
        )

        self.callback.update_locals(locals())
        self.callback.on_train()

    def _evaluate(self):
        self.agent.eval()
        self.envs.train(False)
        mean, std = evaluate_agent(
            agent=self.agent,
            envs=self.envs,
            n_eval_episodes=self.args.n_eval_episodes,
        )
        return mean, std

    def learn(self):

        self._env_reset()
        epoch = 0
        while self.global_step < self.args.total_timesteps:
            epoch += 1
            self.agent.train()
            self.envs.train()
            self._collect_rollout(agent=self.agent, callback=self.callback)
            self._train()

            # Save best model
            if (
                self.args.evaluate_freq is not None
                and epoch % self.args.evaluate_freq == 0
            ):
                mean, std = self._evaluate()
                self.logger.info(f"Mean reward: {mean:.2f} +/- {std: .2f}")
                self.writer.add_scalar("metric/eval_reward", mean, epoch)

                if (self.best_value is None) or (self.best_value < mean):
                    self.best_value = mean
                    self.save_model = True
                else:
                    self.save_model = False

            if self.save_model:
                th.save(self.agent.state_dict(), self.model_path)

    def test(self):
        mean, std = self._evaluate()
        self.logger.info(f"Mean reward: {mean:.2f} +/- {std: .2f}")
