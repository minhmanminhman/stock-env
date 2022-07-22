import gym
from stock_env.envs.single_stock_env import SingleStockEnv
from stock_env.ou_simulator import OUParams, OUSimulator
import pandas as pd
import numpy as np
from typing import Tuple

class SimulatorStockEnv(SingleStockEnv):
    def __init__(
        self, 
        size: int,
        equilibrium_price: float = 50,
        params: OUParams = OUParams,
        random_seed: int = None,
        env_params: dict = {}
    ):
        self.size = size
        self.equilibrium_price = equilibrium_price
        self.random_seed = random_seed
        self.params = params
        self.env_params = env_params
        self.simulator = OUSimulator(
            size=self.size,
            equilibrium_price=self.equilibrium_price,
            seed=self.random_seed,
            params=self.params
        )
        self._simulate_df()
        super().__init__(df=self.df, **self.env_params)
    
    def reset(self) -> Tuple[np.ndarray, float, bool, dict]:
        self._simulate_df()
        return super().reset()
    
    def _simulate_df(self):
        self.df = pd.DataFrame(self.simulator.simulate())