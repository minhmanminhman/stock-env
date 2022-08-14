from dataclasses import dataclass
from xml.sax.handler import property_dom_node
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple
from .base_env import BaseStockEnv
from ..utils import check_col
from gym import spaces
from empyrical import max_drawdown
from ..strategy.trend_strategy import *
import stable_baselines3
# quantity = 1000, buy = 100 -> env chinh lai cai action la buy = 0 -> penalty mua du
# them cac action truoc do 2 weeks

class FeaturesExtractor:
    
    def __init__(self):
        self.strategy = ta.Strategy(
            name="Standard Technical Indicator in various research of automatic stock trading",
            description="SMA MACD RSI CCI ADX Bollinger",
            ta=[
                {"kind": "sma", "length": 10},
                {"kind": "rsi", "length": 14},
                {"kind": "cci", "length": 14},
                {"kind": "adx", "length": 14},
                {"kind": "bbands"},
                {"kind": "macd"},
            ]
        )
        self.required_cols = set('time open high low close volume'.split())
        self.feature_cols = "close SMA_10 RSI_14 CCI_14_0.015 BBL_5_2.0 BBU_5_2.0 MACD_12_26_9".split()
        # indicators + cash + holding
        self.feature_dim = len(self.feature_cols) + 2
    
    def preprocess(self, df):
        check_col(df, self.required_cols)
        df.sort_values(by='time', inplace=True)
        # create indicators
        df.ta.strategy(self.strategy)
        
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)
        df.sort_values(by='time', inplace=True)
        return df[self.feature_cols]

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

class VietnamStockEnv(BaseStockEnv):

    def __init__(
        self,
        df: pd.DataFrame,
        tick_size: float = 0.05,
        lot_size: int = 100,
        max_trade_lot: int = 5,
        max_lot: int = 10,
        kappa: float = 1e-4,
        init_cash: float = 2e4,
        random_seed: int = None,
    ):
        super().__init__(
            df=df,
            lot_size=lot_size,
            max_trade_lot=max_trade_lot,
            max_lot=max_lot,
            init_cash=init_cash,
            random_seed=random_seed
        )
        self.tick_size = tick_size
        self.kappa = kappa
        
        self.df = self._preprocess(self.df)
        self.close = self.df.close
        self._end_tick = self.df.shape[0] - 1
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(2,),
            dtype=np.float64)
        
        # Vietnam market condition
        # can't short stock
        self.min_quantity = 0
        self.position = Position()
    
    @property
    def nav(self):
        price = self.close.iloc[self._current_tick].item()
        return self.position.quantity * price

    @property
    def pnl_from_holding(self):
        price = self.close.iloc[self._current_tick].item()
        start_price = self.close.iloc[self._start_tick].item()
        return price / start_price
    
    def reset(self) -> Tuple[np.ndarray, float, bool, dict]:
        obs = super().reset()
        self.position.reset()
        self.history.update({
            'actions': [],
            'delta_shares': [],
            'delta_vt': [],
            'total_reward': [],
            'total_profit': [],
            'portfolio_value': [],
            'nav': [],
            'cash': [],
            'time': [],
        })
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._current_tick += 1
        self.position.update_position()
        delta_shares = self._decode_action(action)
        
        # enviroment constraint
        modified_delta_shares = self._modify_quantity(delta_shares)
        self.position.transact_trade(modified_delta_shares)
        assert self.position.quantity <= self.max_quantity, f"delta_shares: {modified_delta_shares}, position: {self.position.__dict__}"
        
        # calculate reward
        delta_vt = self._delta_vt(modified_delta_shares)
        step_reward = self._calculate_reward(delta_vt)
        self.total_reward += step_reward
        
        self._update_portfolio(modified_delta_shares)
        
        # always update history last
        info = dict(
            actions = action,
            delta_shares = modified_delta_shares,
            quantity = self.position.quantity,
            delta_vt = delta_vt,
            total_reward = self.total_reward,
            total_profit = self.total_profit,
            portfolio_value = self.portfolio_value,
            nav = self.nav,
            cash = self.cash,
            time = self.df.time.iloc[self._current_tick],
        )
        self._update_history(info)
        return self._get_observation(), step_reward, self._is_done(), info
    
    def _preprocess(self, df):
        required_col = set('time open high low close volume'.split())
        check_col(df, required_col)
        df.sort_values(by='time', inplace=True)
        
        # create indicators
        df['trends'] = trends(df)
        df = trends_confirm_entries(df)
        df.ta.tsignals(df['trends'], append=True)
        
        df.dropna(inplace=True)
        return df

    def _modify_quantity(self, delta_shares: int) -> int:
        """
        modify quantity according to market contraint like T+3 or max quantity
        """
        _delta_shares = delta_shares
        _quantity = self.position.quantity
        if _delta_shares >= 0: # long or hold
            if _quantity + _delta_shares > self.max_quantity:
                delta_shares = max(self.max_quantity - _quantity, 0)
        else: # short
            short_quantity = min(self.position.on_hand, abs(_delta_shares))
            delta_shares = -short_quantity
        return delta_shares

    def _get_observation(self) -> np.ndarray:
        # process window
        # remove_col = set('time open high close low volume'.split())
        # cols = list(set(self.df.columns).difference(remove_col))
        # features = self.df[cols].iloc[self._current_tick].values
        features = self.df[['TS_Trends']].iloc[self._current_tick].values
        
        quantity = self.position.quantity
        obs = np.append(features, quantity).astype(np.float32)
        return obs
    
    def _spread_cost(self, delta_shares: int) -> float:
        return abs(delta_shares) * self.tick_size

    def _impact_cost(self, delta_shares: int) -> float:
        return delta_shares ** 2 * self.tick_size / self.lot_size
    
    def _total_cost(self, delta_shares: int) -> float:
        return self._spread_cost(delta_shares) + self._impact_cost(delta_shares)
    
    def _delta_vt(self, delta_shares: int) -> float:
        prev_price = self.close.iloc[self._current_tick-1].item()
        price = self.close.iloc[self._current_tick].item()
        return self.quantity * (price - prev_price) - self._total_cost(delta_shares)
    
    def _decode_action(self, action: int) -> int:
        return np.take(self.quantity_choice, action)

    def _calculate_reward(self, delta_vt: float) -> float:
        money_penalty = -self.cash * (1 - 0.04/365)
        return delta_vt - 0.5 * self.kappa * (delta_vt ** 2) + money_penalty
    
    # def _calculate_reward(self, delta_vt: float) -> float:
    #     money_penalty = -self.cash * (1 - 0.04/365)
    #     baseline = self.init_cash * self.pnl_from_holding
    #     return ((self.portfolio_value - baseline) + money_penalty)
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        data = self.df.merge(history_df, how='inner', on='time')
        return data
    
    def _is_done(self):
        done =  super()._is_done()
        returns = pd.Series(self.history['portfolio_value']).pct_change()
        maximum_drawdown = max_drawdown(returns)
        return done or (maximum_drawdown < -0.5)

class VietnamStockV2Env(BaseStockEnv):

    def __init__(
        self,
        df: pd.DataFrame,
        tick_size: float = 0.05,
        lot_size: int = 100,
        max_trade_lot: int = 5,
        max_lot: int = 10,
        kappa: float = 1e-4,
        init_cash: float = 2e4,
        random_seed: int = None,
    ):
        super().__init__(
            df=df,
            lot_size=lot_size,
            max_trade_lot=max_trade_lot,
            max_lot=max_lot,
            init_cash=init_cash,
            random_seed=random_seed
        )
        self.tick_size = tick_size
        self.kappa = kappa
        self.feature_extractor = FeaturesExtractor()
        
        self.features, self.df = self._preprocess(self.df)
        self.close = self.df.close
        self._end_tick = self.df.shape[0] - 1
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.feature_extractor.feature_dim,),
            dtype=np.float64)
        
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(1,),
            dtype=np.float32)
        
        # Vietnam market condition
        # can't short stock
        self.min_quantity = 0
        self.position = Position()
    
    @property
    def nav(self):
        price = self.close.iloc[self._current_tick].item()
        return self.position.quantity * price

    @property
    def pnl_from_holding(self):
        price = self.close.iloc[self._current_tick].item()
        start_price = self.close.iloc[self._start_tick].item()
        return price / start_price
    
    @property
    def prev_price(self):
        return self.close.iloc[self._current_tick-1].item()
    
    @property
    def price(self):
        return self.close.iloc[self._current_tick].item()
    
    def reset(self) -> Tuple[np.ndarray, float, bool, dict]:
        obs = super().reset()
        self.position.reset()
        self.history.update({
            'actions': [],
            'delta_shares': [],
            'delta_vt': [],
            'total_reward': [],
            'total_profit': [],
            'portfolio_value': [],
            'nav': [],
            'cash': [],
            'time': [],
            'step_reward': []
        })
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._current_tick += 1
        self.position.update_position()
        delta_shares = self._decode_action(action)
        
        # enviroment constraint
        modified_delta_shares = self._modify_quantity(delta_shares)
        self.position.transact_trade(modified_delta_shares)
        assert self.position.quantity <= self.max_quantity, f"delta_shares: {modified_delta_shares}, position: {self.position.__dict__}"
        
        # calculate reward
        delta_vt = self._delta_vt(modified_delta_shares)
        step_reward = self._calculate_reward(delta_vt)
        self.total_reward += step_reward
        
        self._update_portfolio(modified_delta_shares)
        
        # always update history last
        info = dict(
            actions = action,
            delta_shares = modified_delta_shares,
            quantity = self.position.quantity,
            delta_vt = delta_vt,
            total_reward = self.total_reward,
            total_profit = self.total_profit,
            portfolio_value = self.portfolio_value,
            nav = self.nav,
            cash = self.cash,
            time = self.df.time.iloc[self._current_tick],
            step_reward = step_reward,
        )
        self._update_history(info)
        return self._get_observation(), step_reward, self._is_done(), info
    
    def _preprocess(self, df):
        required_col = set('time open high low close volume'.split())
        check_col(df, required_col)
        df.sort_values(by='time', inplace=True)
        df = df.reset_index(drop=True)
        features = self.feature_extractor.preprocess(df)
        return features, df

    def _modify_quantity(self, delta_shares: int) -> int:
        """
        modify quantity according to market contraint like T+3 or max quantity
        """
        _delta_shares = delta_shares
        _quantity = self.position.quantity
        if _delta_shares >= 0: # long or hold
            if _quantity + _delta_shares > self.max_quantity:
                delta_shares = max(self.max_quantity - _quantity, 0)
            
            _buy_value = self.prev_price * delta_shares + self._total_cost(delta_shares)
            
            while _buy_value > self.cash:
                delta_shares = (int(delta_shares / self.lot_size) - 1) * self.lot_size
                _buy_value = self.prev_price * delta_shares + self._total_cost(delta_shares)
                
        else: # short
            short_quantity = min(self.position.on_hand, abs(_delta_shares))
            delta_shares = -short_quantity
        return delta_shares

    def _get_observation(self) -> np.ndarray:
        features = self.features.iloc[self._current_tick].values
        quantity = self.position.quantity
        obs = np.append(features, quantity).astype(np.float32)
        obs = np.append(obs, self.cash).astype(np.float32)
        return obs
    
    def _spread_cost(self, delta_shares: int) -> float:
        return abs(delta_shares) * self.tick_size

    def _impact_cost(self, delta_shares: int) -> float:
        return delta_shares ** 2 * self.tick_size / self.lot_size
    
    def _total_cost(self, delta_shares: int) -> float:
        return self._spread_cost(delta_shares) + self._impact_cost(delta_shares)
    
    def _delta_vt(self, delta_shares: int) -> float:
        return self.position.quantity * (self.price - self.prev_price) - self._total_cost(delta_shares)
    
    # def _decode_action(self, action: int) -> int:
    #     return np.take(self.quantity_choice, action)
    
    def _decode_action(self, action: float) -> int:
        stock_percent = self.nav / self.portfolio_value
        diff_percent = action - stock_percent
        diff_value = diff_percent * self.portfolio_value
        diff_shares = int((diff_value / self.prev_price) / self.lot_size) * self.lot_size
        return np.clip(diff_shares, self.min_quantity, self.max_quantity)

    def _calculate_reward(self, delta_vt: float) -> float:
        money_penalty = -self.cash * (1 - 0.04/365)
        return delta_vt - 0.5 * self.kappa * (delta_vt ** 2) + money_penalty
    
    # def _calculate_reward(self, delta_vt: float) -> float:
    #     money_penalty = -self.cash * (1 - 0.04/365)
    #     baseline = self.init_cash * self.pnl_from_holding
    #     return ((self.portfolio_value - baseline) + money_penalty)
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        data = self.df.merge(history_df, how='inner', on='time')
        return data
    
    def _is_done(self):
        done =  super()._is_done()
        returns = pd.Series(self.history['portfolio_value']).pct_change()
        maximum_drawdown = max_drawdown(returns)
        return done or (maximum_drawdown < -0.5)