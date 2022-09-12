import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple
from .base_env import BaseVietnamStockEnv
from gym import spaces
from ..feature.feature_extractor import BaseFeaturesExtractor
import mt4_hst
from ..utils import check_col
from .base_env import Position

class RandomStockEnv(BaseVietnamStockEnv):

    def __init__(
        self,
        tickers: list,
        data_folder_path: str,
        feature_extractor: BaseFeaturesExtractor,
        tick_size: float = 0.05,
        lot_size: int = 100,
        init_cash: float = 2e4,
        random_seed: int = None,
        fee: float = 0.001,
    ):
        self.seed(random_seed)
        self.tickers = tickers
        self.data_folder_path = data_folder_path
        # market params
        self.lot_size = lot_size
        self.tick_size = tick_size
        self.fee = fee
        # portfolio params
        self.init_cash = init_cash
        self.position = Position()
        
        # setup data
        self.feature_extractor = feature_extractor
        self.stack_features, self.stack_ohlcv = self._preprocess()
        # reset env need assign these
        self.ohlcv = None
        self.features = None
        
        # obs and action
        self.obs_dim = self.feature_extractor.feature_dim + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        
        # update these variables in reset method
        self._start_tick = None
        self.n_steps = None
        self.cash = None
        self.done = None
        self.history = None        
        self._current_tick = None
        self._end_tick = None
        self._end_episode_tick = None
    
    def reset(self, eval_mode=False) -> Tuple[np.ndarray, float, bool, dict]:
        self._randomize_period(eval_mode=eval_mode)
        self.done = False
        self.n_steps = 0
        self.cash = self.init_cash
        self.position.reset()
        self.history = {
            'quantity': [],
            'actions': [],
            'delta_shares': [],
            'total_profit': [],
            'portfolio_value': [],
            'nav': [],
            'cash': [],
            'time': [],
            'step_reward': [],
            'ticker': []
        }
        return self._get_observation()

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
            ticker = self.episode_ticker
        )
        self._update_history(info)
        return self._get_observation(), step_reward, self._is_done(), info
    
    def _preprocess(self):
        df = pd.DataFrame()
        required_col = set('time open high low close volume'.split())
        for ticker in self.tickers:
            _df = mt4_hst.read_hst(self.data_folder_path + ticker + "1440.hst")
            check_col(_df, required_col)
            _df.sort_values(by='time', inplace=True)
            _df['ticker'] = ticker
            df = pd.concat([df, _df])
        
        grouped_ohlcv = df.groupby('ticker')
        stack_features = grouped_ohlcv.apply(lambda x: self.feature_extractor.preprocess(x))
        
        stack_ohlcv = df.set_index(keys=['ticker', df.index])
        stack_ohlcv, _ = stack_ohlcv.align(stack_features, join='inner', axis=0)
        
        return stack_features, stack_ohlcv

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
        cash_percent = self.cash / self.portfolio_value
        obs = np.append(features, cash_percent).astype(np.float32)
        return obs
    
    def _randomize_period(self, max_timesteps: int = 250, eval_mode: bool = False):
        
        # random choose stock and start period
        self.episode_ticker = np.random.choice(self.tickers, size=1).item()
        self.ohlcv = self.stack_ohlcv.loc[self.episode_ticker]
        self.features = self.stack_features.loc[self.episode_ticker]
        self._end_tick = self.ohlcv.shape[0] - 1
        
        if eval_mode:
            start_tick = 0
            end_tick = self._end_tick
        else:
            # # method 1
            # self._current_tick = np.random.randint(self._start_tick, self._end_tick)
            # # train 500 timesteps or until the end
            # self._end_episode_tick = min(self._current_tick + max_timesteps, self._end_tick)
            
            # method 2
            start_idxes = np.arange(start=0, stop=self.ohlcv.shape[0], step=max_timesteps)
            start_tick = np.random.choice(start_idxes)
            end_tick = min(start_tick + max_timesteps, self._end_tick)
        
        self._start_tick = self._current_tick = start_tick
        self._end_episode_tick = end_tick
    
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
        cum_return = (self.portfolio_value + eps) / (self.init_cash + eps)
        # compare with holding
        cum_return_from_holding = self.close / self.ohlcv.close.iloc[self._start_tick].item()
        diff = cum_return - cum_return_from_holding
        
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
        return reward
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        history_df = history_df.astype({'time':'datetime64[ns]'})
        data = self.ohlcv.merge(history_df, how='inner', on='time')
        data = data.join(self.features)
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