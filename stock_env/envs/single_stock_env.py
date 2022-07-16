import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
from typing import Tuple

class SingleStockEnv(gym.Env):
    """
    Stock env in paper: Machine Learning for trading
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3015609
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        tick_size: float = 0.1,
        lot_size: int = 100,
        max_trade_lot: int = 5,
        max_lot: int = 10,
        kappa: float = 1e-4,
        init_cash: float = 5e4,
    ):

        self.seed()
        
        # data
        self.df = df
        
        # market params
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_trade_lot = max_trade_lot
        self.max_quantity = max_lot * lot_size
        self.min_quantity = - max_lot * lot_size
        self.kappa = kappa
        
        # portfolio params
        self.init_cash = init_cash
        self.cash = None
        self.nav = None
        self.shares = self.lot_size * np.arange(-max_trade_lot, max_trade_lot+1)

        # spaces
        self.n_action = len(self.shares)
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # episode
        self._start_tick = 0
        self._end_tick = self.df.shape[0] - 1
        self._current_tick = None
        self.done = None
        self.total_reward = None
        # self.total_profit = None
        self.quantity = None
        self.history = None
    
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
        # self.total_profit = 0  # unit
        self.cash = self.init_cash
        self.nav = 0
        self.quantity = 0
        self.history = {
            'actions': [],
            'delta_shares': [],
            'quantity': [],
            'delta_vt': [],
            'total_reward': [],
            'total_profit': [],
            'portfolio_value': [],
            'nav': [],
            'cash': [],
        }
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._current_tick += 1
        delta_shares = self._decode_action(action)
        
        # enviroment constraint
        quantity = self._get_prev_quantity() + delta_shares
        self.quantity = np.clip(quantity, self.min_quantity, self.max_quantity)
        delta_shares = self.quantity - self._get_prev_quantity()
        
        # print(f"{self._get_prev_quantity()} + {delta_shares} = {self.quantity}")
        
        # calculate reward
        delta_vt = self._delta_vt(delta_shares)
        step_reward = self._calculate_reward(delta_vt)
        self.total_reward += step_reward
        
        self._update_portfolio(delta_shares)
        
        # always update history last
        info = dict(
            actions = action,
            delta_shares = delta_shares,
            quantity = self.quantity,
            delta_vt = delta_vt,
            total_reward = self.total_reward,
            total_profit = self.total_profit,
            portfolio_value = self.portfolio_value,
            nav = self.nav,
            cash = self.cash,
        )
        self._update_history(info)
        # print(f"next_obs: {self._get_observation()}")
        return self._get_observation(), step_reward, self._is_done(), info

    def _get_observation(self) -> np.ndarray:
        # process window
        price = self.df.iloc[self._current_tick].to_numpy()
        quantity = self._get_prev_quantity()

        return np.concatenate([price, np.asarray([quantity])])
    
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
    
    def spread_cost(self, delta_shares: int) -> float:
        return abs(delta_shares) * self.tick_size

    def impact_cost(self, delta_shares: int) -> float:
        return delta_shares ** 2 * self.tick_size / self.lot_size
    
    def total_cost(self, delta_shares: int) -> float:
        return self.spread_cost(delta_shares) + self.impact_cost(delta_shares)
    
    def _delta_vt(self, delta_shares: int) -> float:
        prev_price = self.df.iloc[self._current_tick-1].item()
        price = self.df.iloc[self._current_tick].item()
        return self.quantity * (price - prev_price) - self.total_cost(delta_shares)
    
    def _update_portfolio(self, delta_shares: int) -> float:
        prev_price = self.df.iloc[self._current_tick-1].item()
        price = self.df.iloc[self._current_tick].item()
        self.nav = self.quantity * price
        self.cash -= (delta_shares * prev_price + self.total_cost(delta_shares))
    
    def _is_done(self):
        if (self._current_tick == self._end_tick) \
            or (self.portfolio_value <= 0):
            return True
        return False
    
    def _decode_action(self, action: int) -> int:
        return np.take(self.shares, action)

    def _calculate_reward(self, delta_vt: float) -> float:
        return delta_vt - 0.5 * self.kappa * (delta_vt ** 2)