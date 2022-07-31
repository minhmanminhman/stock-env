from dataclasses import dataclass
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple
from .base_env import BaseStockEnv
from ..utils import check_col
from gym import spaces
from empyrical import max_drawdown

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
            shape=(14,),
            dtype=np.float64)
        
        # Vietnam market condition
        # can't short stock
        self.min_quantity = 0
        self.position = Position()
    
    @property
    def nav(self):
        price = self.close.iloc[self._current_tick].item()
        return self.position.quantity * price
    
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
        df.ta.rsi(length=20, append=True)
        df.ta.natr(length=20, scalar=1, append=True)
        df.ta.log_return(length=5, append=True)
        df.ta.log_return(length=20, append=True)
        df.ta.percent_return(length=5, append=True)
        df.ta.percent_return(length=20, append=True)

        # trend setup
        df['close>sma50'] = (df['close'] > df.ta.sma(50)).astype(int)
        df['close>sma100'] = (df['close'] > df.ta.sma(100)).astype(int)
        df['close>sma200'] = (df['close'] > df.ta.sma(200)).astype(int)
        df['ema10>ema20'] = (df.ta.ema(10) > df.ta.ema(20)).astype(int)
        donchian_20 = ta.donchian(df['high'], df['close'], lower_length=20, upper_length=20)
        donchian_50 = ta.donchian(df['high'], df['close'], lower_length=50, upper_length=50)
        df['higher_low'] = (donchian_20['DCL_20_20'] > donchian_50['DCL_50_50']).astype(int)
        df['breakout'] = (df['close'] > donchian_20['DCU_20_20'].shift(1)).astype(int)

        # volume confirm
        df['volume_breakout'] = (df['volume'] > ta.sma(df['volume'], 20)).astype(int)
        
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
        remove_col = set('time open high close low volume'.split())
        cols = list(set(self.df.columns).difference(remove_col))
        features = self.df[cols].iloc[self._current_tick].values
        
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
        return delta_vt - 0.5 * self.kappa * (delta_vt ** 2)
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        data = self.df.merge(history_df, how='inner', on='time')
        return data
    
    def _is_done(self):
        done =  super()._is_done()
        returns = pd.Series(self.history['portfolio_value']).pct_change()
        maximum_drawdown = max_drawdown(returns)
        return done or (maximum_drawdown > 0.2)