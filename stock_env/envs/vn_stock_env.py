from dataclasses import dataclass
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple
from .base_env import BaseStockEnv
from ..utils import check_col
from gym import spaces
from empyrical import max_drawdown

# Create your own Custom Strategy
TrendStrategy = ta.Strategy(
    name="Trend Signal Strategy",
    description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
    ta=[
        {"kind": "percent_return", "length": 1},
        {"kind": "sma", "length": 50},
        {"kind": "ema", "length": 10},
        {"kind": "ema", "length": 20},
        {"kind": "donchian", "lower_length": 10, "upper_length": 10},
        {"kind": "donchian", "lower_length": 20, "upper_length": 20},
        {"kind": "donchian", "lower_length": 50, "upper_length": 50},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
    ]
)

# v2
def trends(df: pd.DataFrame):
    # create neccesary indicators
    df.ta.strategy(TrendStrategy)
    
    # condition
    trend_cond = (
        (df['close'] > df['SMA_50']).astype(int)
        + (df['EMA_10'] > df['EMA_20']).astype(int)
        + (df['DCL_10_10'] > df['DCL_50_50']).astype(int)
    ) >= 3
    
    return trend_cond

def trends_confirm_entries(df: pd.DataFrame):
    volume_breakout = (df['volume'] >= 1.25 * df['VOLUME_SMA_20']).astype(int)
    
    # candle pattern
    entry_pattern = df.ta.cdl_pattern(
        name=["closingmarubozu", "marubozu", "engulfing", "longline"], 
        scalar=1).sum(axis=1).astype(int)
    
    exit_pattern = df.ta.cdl_pattern(
        name=[
            # "doji", "dojistar", "dragonflydoji", "eveningdojistar", "invertedhammer", "eveningstar", "gravestonedoji", "hangingman", 
            "closingmarubozu", "marubozu", "engulfing", "longline"
            # "3blackcrows", "longleggeddoji", "shootingstar", "spinningtop"
        ], 
        scalar=1).sum(axis=1).astype(int)
    
    df['entry'] = ((entry_pattern > 0) * volume_breakout * (df['close'] > df['DCU_10_10'].shift())).astype(bool).astype(int)
    df['exit'] = (
        (exit_pattern < 0) 
        # * volume_breakout 
        # * (df['close'] < df['DCL_10_10'].shift())
    ).astype(bool).astype(int)
    
    def modify_entry_exit(df):
        df['trends_'] = np.nan
        
        try:
            entry_idx = df[df['entry'] == 1].index[0]
            df['trends_'].loc[entry_idx] = 1
            
            try:
                exit_idx = df[df['exit'] == 1].index[-1]
                if entry_idx <= exit_idx:
                    df['trends_'].loc[exit_idx] = 1
            except:
                pass
            finally:
                df['trends_'] = df['trends_'].ffill()
        except IndexError:
            df['trends_'] = df['trends_'].fillna(0)
        
        try:
            df['trends_'] = df['trends_'].interpolate(limit_area='inside', method='nearest')
        except ValueError:
            df['trends_'] = df['trends_'].fillna(0)
        return df['trends_'].fillna(0) * df['trends']

    df['original_trends'] = df['trends']
    df['trends'] = df.groupby((df.trends != df.trends.shift()).cumsum()).apply(modify_entry_exit).to_list()
    return df

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
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        data = self.df.merge(history_df, how='inner', on='time')
        return data
    
    def _is_done(self):
        done =  super()._is_done()
        returns = pd.Series(self.history['portfolio_value']).pct_change()
        maximum_drawdown = max_drawdown(returns)
        return done or (maximum_drawdown > 0.2)