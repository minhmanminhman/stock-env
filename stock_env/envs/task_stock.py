from ..data_loader import BaseTaskLoader
from .random_stock import RandomStockEnv


class TaskStockEnv(RandomStockEnv):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data_loader: BaseTaskLoader,
        lot_size: int = 100,
        init_cash: float = 2e4,
        random_seed: int = None,
        fee: float = 0.001,
    ):
        super().__init__(
            data_loader=data_loader,
            lot_size=lot_size,
            init_cash=init_cash,
            random_seed=random_seed,
            fee=fee,
        )

    def reset_task(self, task: str) -> None:
        self.data_loader.reset_task(task)

    def sample_task(self) -> str:
        return self.data_loader.sample_task()

    def train(self, mode: bool = True):
        self.data_loader.train(mode)
