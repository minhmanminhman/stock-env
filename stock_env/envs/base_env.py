from abc import abstractmethod
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
from ..utils import check_col

@dataclass
class Position:
    t0_quantity: int = 0
    t1_quantity: int = 0
    t2_quantity: int = 0
    on_hand: int = 0
    
    @property
    def quantity(self):
        quantity = self.t0_quantity + self.t1_quantity + self.t2_quantity + self.on_hand
        return quantity
    
    def update_position(self):
        self.on_hand += self.t2_quantity
        self.t2_quantity = self.t1_quantity
        self.t1_quantity = self.t0_quantity
        self.t0_quantity = 0
    
    def transact_trade(self, delta_shares):
        if delta_shares >= 0: # long or hold
            self.t0_quantity = delta_shares
        else: # short
            self.on_hand = self.on_hand + delta_shares
            assert self.on_hand >= 0
    
    def reset(self):
        self.t0_quantity = self.t1_quantity = self.t2_quantity = self.on_hand = 0

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

class BaseVietnamStockEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        init_cash: float = 2e4,
        random_seed: int = None,
        ticker: str = None,
    ):
        self.seed(random_seed)
        
        # data
        required_col = set('time open high low close volume'.split())
        check_col(df, required_col)
        df = df.sort_values(by='time')
        self.df = df
        self.close = df.close
        self.ticker = ticker
        # portfolio params
        self.init_cash = init_cash
        self.cash = None
        self.position = Position()

        # episode
        self._start_tick = 0
        self._end_tick = self.df.shape[0] - 1
        self._current_tick = None
        self.done = None
        self.total_reward = None
        self.history = None
    
    @property
    def portfolio_value(self):
        return self.nav + self.cash
    
    @property
    def total_profit(self):
        return self.portfolio_value - self.init_cash
    
    @property
    def nav(self):
        return self.position.quantity * self.price

    @property
    def prev_price(self):
        return self.close.iloc[self._current_tick-1].item()
    
    @property
    def price(self):
        return self.close.iloc[self._current_tick].item()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, current_tick='random') -> Tuple[np.ndarray, float, bool, dict]:
        if current_tick == 'random':
            self._current_tick = np.random.randint(self._start_tick, self._end_tick)
        elif isinstance(current_tick, int):
            self._current_tick = current_tick
        else:
            raise NotImplementedError
        
        self.done = False
        self.total_reward = 0
        self.cash = self.init_cash
        self.position.reset()
        self.history = {'quantity': []}
        return self._get_observation()

    def _update_history(self, info: dict) -> None:
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)
    
    def _is_done(self):
        if (self._current_tick == self._end_tick) \
            or (self.portfolio_value <= 0):
            return True
        return False
    
    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _preprocess(self, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError