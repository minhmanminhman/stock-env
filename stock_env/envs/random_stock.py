import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple
from .base_env import BaseVietnamStockEnv
from gym import spaces
from ..feature.feature_extractor import BaseFeaturesExtractor

class RandomStockEnv(BaseVietnamStockEnv):

    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: BaseFeaturesExtractor,
        tick_size: float = 0.05,
        lot_size: int = 100,
        init_cash: float = 2e4,
        random_seed: int = None,
        ticker: str = None,
        fee: float = 0.001,
    ):
        super().__init__(
            df=df,
            init_cash=init_cash,
            random_seed=random_seed,
            ticker=ticker,
        )
        
        # market params
        self.lot_size = lot_size
        self.tick_size = tick_size
        self.fee = fee
        
        # setup data
        self.feature_extractor = feature_extractor
        self.features, self.ohlcv = self._preprocess(self.ohlcv)
        self._end_tick = self.ohlcv.shape[0] - 1
        
        # obs and action
        self.obs_dim = self.feature_extractor.feature_dim + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, float, bool, dict]:
        obs = super().reset(**kwargs)
        self.history.update({
            'actions': [],
            'delta_shares': [],
            'total_profit': [],
            'portfolio_value': [],
            'nav': [],
            'cash': [],
            'time': [],
            'step_reward': []
        })
        return obs

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        self._current_tick += 1
        self.n_steps += 1
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
            total_profit = self.total_profit,
            portfolio_value = self.portfolio_value,
            nav = self.nav,
            cash = self.cash,
            time = self.ohlcv.time.iloc[self._current_tick],
            step_reward = step_reward,
        )
        self._update_history(info)
        return self._get_observation(), step_reward, self._is_done(), info
    
    def _preprocess(self, df):
        features = self.feature_extractor.preprocess(df)
        df, _ = df.align(features, join='inner', axis=0)
        return features, df

    def _modify_quantity(self, delta_shares: int) -> int:
        """
        modify quantity according to market contraint like T+3 or max quantity
        """
        _delta_shares = delta_shares
        if _delta_shares >= 0: # long or hold
            _buy_value = self.open * delta_shares + self._total_cost(delta_shares)
            
            while _buy_value > self.cash:
                delta_shares = (int(delta_shares / self.lot_size) - 1) * self.lot_size
                _buy_value = self.open * delta_shares + self._total_cost(delta_shares)
                
        else: # short
            short_quantity = min(self.position.on_hand, abs(_delta_shares))
            delta_shares = -short_quantity
        return delta_shares

    def _get_observation(self) -> np.ndarray:
        features = self.features.iloc[self._current_tick].values
        quantity = self.position.quantity
        obs = np.append(features, quantity).astype(np.float32)
        return obs
    
    def _total_cost(self, delta_shares: int) -> float:
        if delta_shares >= 0:
            cost = delta_shares * self.open * self.fee
        else:
            # selling stock has PIT fee = 0.1%
            cost = abs(delta_shares) * self.open * (self.fee + 0.001)
        return cost
    
    def _decode_action(self, action: float) -> int:
        target_cash = action * self.portfolio_value
        diff_value = self.cash - target_cash
        # diff_value > 0 -> buy shares
        # diff_value < 0 -> sell shares
        diff_shares = int((diff_value / self.open) / self.lot_size) * self.lot_size
        return diff_shares
    
    def _calculate_reward(self, *args, **kwargs) -> float:
        eps = 1e-8
        cum_log_return = np.log((self.portfolio_value + eps) / (self.init_cash + eps))
        reward = cum_log_return / self.n_steps
        return reward
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        history_df = history_df.astype({'time':'datetime64[ns]'})
        data = self.ohlcv.merge(history_df, how='inner', on='time')
        return data

    def _update_portfolio(self, delta_shares: int) -> float:
        # buy/sell at open price
        self.cash -= (delta_shares * self.open + self._total_cost(delta_shares))
        assert self.cash >= 0
    
    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = 0, 1
        return low + (0.5 * (scaled_action + 1.0) * (high - low))