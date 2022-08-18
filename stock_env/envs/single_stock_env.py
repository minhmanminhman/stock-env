import numpy as np
import pandas as pd
from typing import Tuple
from .base_env import BaseStockEnv

class SingleStockEnv(BaseStockEnv):
    """
    Stock env in paper: Machine Learning for trading
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3015609
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tick_size: float = 0.1,
        lot_size: int = 100,
        max_trade_lot: int = 5,
        max_lot: int = 10,
        kappa: float = 1e-4,
        init_cash: float = 2e4,
        random_seed: int = None,
    ):
        super(SingleStockEnv, self).__init__(
            df=df,
            lot_size=lot_size,
            max_trade_lot=max_trade_lot,
            max_lot=max_lot,
            init_cash=init_cash,
            random_seed=random_seed
        )
        self.seed(random_seed)
        self.tick_size = tick_size
        self.kappa = kappa
    
    def reset(self) -> Tuple[np.ndarray, float, bool, dict]:
        obs = super(SingleStockEnv, self).reset()
        self.history.update({
            'actions': [],
            'delta_shares': [],
            'delta_vt': [],
            'total_reward': [],
            'total_profit': [],
            'portfolio_value': [],
            'nav': [],
            'cash': [],
        })
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._current_tick += 1
        delta_shares = self._decode_action(action)
        
        # enviroment constraint
        quantity = self._get_prev_quantity() + delta_shares
        self.quantity = np.clip(quantity, self.min_quantity, self.max_quantity)
        delta_shares = self.quantity - self._get_prev_quantity()
        
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
        return self._get_observation(), step_reward, self._is_done(), info

    def _get_observation(self) -> np.ndarray:
        # process window
        price = self.df.iloc[self._current_tick].item()
        quantity = self._get_prev_quantity()

        return np.asarray([float(price), float(quantity)])
    
    def _total_cost(self, delta_shares: int) -> float:
        spread_cost = abs(delta_shares) * self.tick_size
        impact_cost = delta_shares ** 2 * self.tick_size / self.lot_size
        return spread_cost + impact_cost
    
    def _delta_vt(self, delta_shares: int) -> float:
        prev_price = self.df.iloc[self._current_tick-1].item()
        price = self.df.iloc[self._current_tick].item()
        return self.quantity * (price - prev_price) - self._total_cost(delta_shares)
    
    def _decode_action(self, action: int) -> int:
        return np.take(self.quantity_choice, action)

    def _calculate_reward(self, delta_vt: float) -> float:
        return delta_vt - 0.5 * self.kappa * (delta_vt ** 2)