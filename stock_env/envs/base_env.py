from abc import abstractmethod
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
from typing import Tuple

class BaseStockEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        lot_size: int = 100,
        max_trade_lot: int = 5,
        max_lot: int = 10,
        init_cash: float = 2e4,
        random_seed: int = None,
    ):
        self.seed(random_seed)
        
        # data
        self.df = df
        self.close = df
        
        # market params
        self.lot_size = lot_size
        self.max_trade_lot = max_trade_lot
        self.max_quantity = max_lot * lot_size
        self.min_quantity = -max_lot * lot_size
        
        # portfolio params
        self.init_cash = init_cash
        self.cash = None
        self.quantity_choice = self.lot_size * np.arange(-max_trade_lot, max_trade_lot+1)

        # episode
        self._current_tick = self._start_tick = 0
        self._end_tick = self.df.shape[0] - 1
        self._current_tick = None
        self.done = None
        self.total_reward = None
        self.quantity = None
        self.history = None
        
        # spaces
        self.n_action = len(self.quantity_choice)
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(2,), # [price, quantity]
            dtype=np.float64
        )
    
    @property
    def nav(self):
        price = self.close.iloc[self._current_tick].item()
        return self.quantity * price
    
    @property
    def portfolio_value(self):
        return self.nav + self.cash
    
    @property
    def total_profit(self):
        return self.portfolio_value - self.init_cash

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[np.ndarray, float, bool, dict]:
        self._current_tick = self._start_tick
        self.done = False
        self.total_reward = 0
        self.cash = self.init_cash
        self.quantity = 0
        self.history = {'quantity': []}
        return self._get_observation()
    
    def _get_prev_quantity(self) -> int:
        try:
            quantity = self.history['quantity'][-1]
        except IndexError:
            quantity = 0
        return quantity

    def _update_history(self, info: dict) -> None:
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)
    
    def _update_portfolio(self, delta_shares: int) -> float:
        prev_price = self.close.iloc[self._current_tick-1].item()
        # buy/sell at close price
        self.cash -= (delta_shares * prev_price + self._total_cost(delta_shares))
        assert self.cash >= 0

    def _is_done(self):
        if (self._current_tick == self._end_tick) \
            or (self.portfolio_value <= 0):
            return True
        return False
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _total_cost(self) -> float:
        raise NotImplementedError