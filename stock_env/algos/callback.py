import numpy as np
from abc import ABC, abstractmethod


class BaseCallback(ABC):
    def __init__(self):
        self.locals = {}
        self.globals = {}

    @abstractmethod
    def init_callback(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def on_dones(self, *args, **kwargs):
        raise NotImplementedError

    def update_locals(self, locals_, globals_):
        self.locals.update(locals_)
        self.globals.update(globals_)


class PPOLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def init_callback(self, logger, writer):
        self.logger = logger
        self.writer = writer

    def on_dones(self):
        global_step = self.locals["self"].global_step
        mean_reward = self.locals["mean_reward"]
        is_training = self.locals["self"].agent.training
        self.logger.info(
            f"global_step={global_step}, episodic_return={mean_reward :.2f}"
        )
        if is_training:
            self.writer.add_scalar("metric/train_reward", mean_reward, global_step)
        else:
            self.writer.add_scalar("metric/eval_reward", mean_reward, global_step)

    def on_train(self):
        global_step = self.locals["self"].global_step
        loss = self.locals["loss"]
        entropy_losses = self.locals["entropy_losses"]
        pg_losses = self.locals["pg_losses"]
        value_losses = self.locals["value_losses"]
        explained_var = self.locals["explained_var"]
        total_norm = self.locals["total_norm"]

        self.writer.add_scalar(
            "train/entropy_loss", np.mean(entropy_losses), global_step
        )
        self.writer.add_scalar(
            "train/policy_gradient_loss", np.mean(pg_losses), global_step
        )
        self.writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
        self.writer.add_scalar("train/loss", loss.item(), global_step)
        self.writer.add_scalar("train/explained_variance", explained_var, global_step)
        self.writer.add_scalar("train/total_norm", total_norm.item(), global_step)


class MAMLLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def init_callback(self, logger, writer):
        self.logger = logger
        self.writer = writer

    def on_dones(self):
        global_step = self.locals["self"].global_step
        mean_reward = self.locals["mean_reward"]
        is_training = self.locals["self"].agent.training
        epoch = self.locals["self"].epoch

        if is_training:
            self.writer.add_scalar("metric/train_reward", mean_reward, global_step)
        else:
            self.writer.add_scalar("metric/eval_reward", mean_reward, global_step)
            self.logger.info(
                f"global_step={global_step}, episodic_return={mean_reward :.2f}, epoch={epoch}"
            )

    def on_inner_train(self):
        global_step = self.locals["self"].global_step
        inner_loss = self.locals["mean_train_task_loss"]
        outer_loss = self.locals["outer_loss"].item()

        self.writer.add_scalar("inner_train/inner_loss", inner_loss, global_step)
        self.writer.add_scalar("inner_train/outer_loss", outer_loss, global_step)
        self.logger.info(
            f"global_step={global_step}, inner_loss={inner_loss :.2f}, outer_loss={outer_loss :.2f}"
        )

    def on_outer_train(self):
        _self = self.locals["self"]
        epoch = _self.epoch
        outer_loss = self.locals["outer_loss"].item()
        total_norm = self.locals.get("total_norm", 0)

        self.writer.add_scalar("outer_train/outer_loss", outer_loss, epoch)
        self.writer.add_scalar("outer_train/total_norm", total_norm, epoch)
        self.logger.info(f"epoch={epoch}, outer_loss={outer_loss :.2f}")

        if _self.run_name.startswith("mamlpp"):
            self.logger.info(f"lrs={[lr.item() for lr in _self.meta_agent.lrs]}")
