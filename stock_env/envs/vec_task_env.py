from gymnasium.vector import SyncVectorEnv
from typing import List
import numpy as np


class MetaVectorStockEnv(SyncVectorEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)

    def reset_task(self, task: str):
        for env in self.envs:
            env.reset_task(task)

    def sample_task(self, num_tasks) -> List[str]:
        tasks = self.envs[0].data_loader.tickers
        assert num_tasks <= len(
            tasks
        ), f"num_tasks {num_tasks} > len(tasks) {len(tasks)}"
        return np.random.choice(tasks, size=num_tasks, replace=False)

    def train(self, mode=True):
        for env in self.envs:
            env.data_loader.train(mode)


class MetaVectorEnv(SyncVectorEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)

    def reset_task(self, task: str):
        for env in self.envs:
            env.reset_task(task)

    def sample_task(self, num_tasks) -> List[str]:
        return self.envs[0].sample_task(num_tasks)

    def train(self, mode=True):
        pass
