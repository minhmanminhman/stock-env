import numpy as np
import pandas as pd
from typing import Tuple
from gym import spaces
from gym import register
from .base_env import BaseVietnamStockEnv
from .base_env import Position
from ..data_loader import BaseDataLoader
from empyrical import max_drawdown

class RandomStockEnv(BaseVietnamStockEnv):

    def __init__(
        self,
        data_loader: BaseDataLoader,
        lot_size: int = 100,
        init_cash: float = 2e4,
        random_seed: int = None,
        fee: float = 0.001,
    ):
        super().__init__(
            data_loader=data_loader,
            init_cash=init_cash,
            random_seed=random_seed
        )
        # market params
        self.lot_size = lot_size
        self.fee = fee
        self.position = Position()
        # obs and action
        self.obs_dim = self.data_loader.feature_dim + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        
        # update these variables in reset method
        self.n_steps = None
        self.cash = None
        self.done = None
        self.history = None
    
    @property
    def portfolio_value(self):
        return self.nav + self.cash
    
    @property
    def nav(self):
        return self.position.quantity * self.close_price

    @property
    def open_price(self):
        return self.data_loader.current_ohlcv.open.item()
    
    @property
    def close_price(self):
        return self.data_loader.current_ohlcv.close.item()

    def reset(self) -> Tuple[np.ndarray, float, bool, dict]:
        self.position.reset()
        self.done = False
        self.n_steps = 0
        self.cash = self.init_cash
        self.history = {
            'quantity': [],
            'actions': [],
            'delta_shares': [],
            'portfolio_value': [],
            'nav': [],
            'cash': [],
            'time': [],
            'step_reward': [],
        }
        features = self.data_loader.reset()
        self.start_close_price = self.data_loader.current_ohlcv.close.item()
        return self._get_observation(features)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        self.n_steps += 1
        features = self.data_loader.step()
        self.position.update_position()
        action = self.unscale_action(action)
        delta_shares = self._decode_action(action)
        
        # enviroment constraint
        modified_delta_shares = self._modify_quantity(delta_shares)
        self.position.transact_trade(modified_delta_shares)
        self._update_portfolio(modified_delta_shares)
        step_reward = self._calculate_reward()
        
        # always update history last
        info = dict(
            actions = action.item(),
            delta_shares = modified_delta_shares,
            quantity = self.position.quantity,
            portfolio_value = self.portfolio_value,
            nav = self.nav,
            cash = self.cash,
            time = self.data_loader.current_ohlcv.time,
            step_reward = step_reward,
        )
        self._update_history(info)
        return self._get_observation(features), step_reward, self._is_done(), info

    def _modify_quantity(self, delta_shares: int) -> int:
        """
        modify quantity according to market contraint like T+3 or max quantity
        """
        _delta_shares = delta_shares
        if _delta_shares >= 0: # long or hold
            _buy_value = self.open_price * delta_shares + self._total_cost(delta_shares)
            
            while _buy_value > self.cash:
                delta_shares = (int(delta_shares / self.lot_size) - 1) * self.lot_size
                _buy_value = self.open_price * delta_shares + self._total_cost(delta_shares)
                
        else: # short
            short_quantity = min(self.position.on_hand, abs(_delta_shares))
            delta_shares = -short_quantity
        return delta_shares

    def _get_observation(self, features: np.ndarray) -> np.ndarray:
        cash_percent = self.cash / self.portfolio_value
        obs = np.append(features, cash_percent).astype(np.float32)
        return obs
    
    def _total_cost(self, delta_shares: int) -> float:
        if delta_shares >= 0:
            cost = delta_shares * self.open_price * self.fee
        else:
            # selling stock has PIT fee = 0.1%
            cost = abs(delta_shares) * self.open_price * (self.fee + 0.001)
        return cost
    
    def _decode_action(self, action: float) -> int:
        target_cash = action * self.portfolio_value
        diff_value = self.cash - target_cash
        # diff_value > 0 -> buy shares
        # diff_value < 0 -> sell shares
        diff_shares = int((diff_value / self.open_price) / self.lot_size) * self.lot_size
        return diff_shares
    
    def _calculate_reward(self, *args, **kwargs) -> float:
        eps = 1e-8
        cum_return = (self.portfolio_value + eps) / (self.init_cash + eps)
        # compare with holding
        cum_return_from_holding = self.close_price / self.start_close_price
        diff = cum_return - cum_return_from_holding
        returns = pd.Series(self.history['portfolio_value'][-50:]).pct_change()
        max_dd = np.abs(max_drawdown(returns))
        max_dd = max_dd if not np.isnan(max_dd) else 0.
        
        # method 1
        # reward = diff / self.n_steps
        
        # method 2
        if (diff > 0) and (cum_return > 0):
            reward = 2
        elif (diff > 0) and (cum_return < 0):
            reward = 1
        elif (diff > 0) and (cum_return == 0):
            reward = 0
        else:
            reward = -2
        output = reward - (max_dd / self.n_steps)
        return output
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        history_df['ticker'] = self.data_loader.ticker
        # history_df = history_df.astype({'time':'datetime64[ns]'})
        data = self.data_loader.ohlcv.merge(history_df, how='inner', on='time')
        data = data.join(self.data_loader.features)
        return data

    def _update_portfolio(self, delta_shares: int) -> float:
        # buy/sell at open price
        self.cash -= (delta_shares * self.open_price + self._total_cost(delta_shares))
        assert self.cash >= 0
        
    def _is_done(self):
        if self.data_loader.is_done \
            or (self.portfolio_value <= 0):
            return True
        return False
    
    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = 0, 1
        return low + (0.5 * (scaled_action + 1.0) * (high - low))