import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple
from .base_env import BaseVietnamStockEnv
from gym import spaces
from ..feature.feature_extractor import BaseFeaturesExtractor

#TODO: quantity = 1000, buy = 100 -> env chinh lai cai action la buy = 0 -> penalty mua du
#TODO: them cac action truoc do 2 weeks

class VietnamStockEnv(BaseVietnamStockEnv):

    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: BaseFeaturesExtractor,
        tick_size: float = 0.05,
        lot_size: int = 100,
        max_trade_lot: int = 5,
        max_lot: int = 10,
        kappa: float = 1e-4,
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
        self.max_trade_lot = max_trade_lot
        self.max_quantity = max_lot * lot_size
        self.quantity_choice = self.lot_size * np.arange(-max_trade_lot, max_trade_lot+1)
        self.tick_size = tick_size
        self.kappa = kappa
        self.fee = fee
        self.feature_extractor = feature_extractor
        
        # setup data
        self.features, self.df = self._preprocess(self.df)
        self.close = self.df.close
        self._end_tick = self.df.shape[0] - 1
        self.ticker = ticker
        
        # obs and action
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.feature_extractor.feature_dim + 1,),
            dtype=np.float64)
        self.action_space = spaces.Discrete(len(self.quantity_choice))
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, float, bool, dict]:
        obs = super().reset(**kwargs)
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
        action = self.unscale_action(action)
        delta_shares = self._decode_action(action)
        
        # enviroment constraint
        modified_delta_shares = self._modify_quantity(delta_shares)
        trade = {
            'delta_shares': modified_delta_shares,
            'avg_price': self.prev_price
        }
        self.position.transact_trade(trade)
        assert self.position.quantity <= self.max_quantity, f"delta_shares: {modified_delta_shares}, position: {self.position.__dict__}"
        self._update_portfolio(modified_delta_shares)
        
        # calculate reward
        delta_vt = self._delta_vt(modified_delta_shares)
        step_reward = self._calculate_reward(delta_vt)
        self.total_reward += step_reward
        
        
        # always update history last
        info = dict(
            actions = action.item(),
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
        features = self.feature_extractor.preprocess(df)
        df, _ = df.align(features, join='inner', axis=0)
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
        return obs
    
    def _total_cost(self, delta_shares: int) -> float:
        if delta_shares >= 0:
            cost = delta_shares * self.prev_price * self.fee
        else:
            # selling stock has PIT fee = 0.1%
            cost = abs(delta_shares) * self.prev_price * (self.fee + 0.001)
        return cost
    
    def _delta_vt(self, delta_shares: int) -> float:
        return self.position.quantity * (self.price - self.prev_price) - self._total_cost(delta_shares)
    
    def _decode_action(self, action: int) -> int:
        return np.take(self.quantity_choice, action)
    
    def _calculate_reward(self, delta_vt: float) -> float:
        return delta_vt
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        history_df = history_df.astype({'time':'datetime64[ns]'})
        data = self.df.merge(history_df, how='inner', on='time')
        return data

    def _update_portfolio(self, delta_shares: int) -> float:
        # buy/sell at close price
        self.cash -= (delta_shares * self.prev_price + self._total_cost(delta_shares))
        assert self.cash >= 0
        
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
            shape=(self.feature_extractor.feature_dim + 2,),
            dtype=np.float64)
        
        self.action_space = spaces.Box(
            low=-1, 
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
    def invested_value(self):
        return self.cash + self.position.invested_value
    
    @property
    def prev_price(self):
        return self.close.iloc[self._current_tick-1].item()
    
    @property
    def price(self):
        return self.close.iloc[self._current_tick].item()
    
    def reset(self) -> Tuple[np.ndarray, float, bool, dict]:
        obs = super().reset()
        self._current_tick = np.random.randint(self._start_tick, self._end_tick)
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
            'step_reward': [],
            'invested_value': [],
            'avg_price': [],
        })
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._current_tick += 1
        self.position.update_position()
        action = self.unscale_action(action)
        delta_shares = self._decode_action(action)
        
        # enviroment constraint
        modified_delta_shares = self._modify_quantity(delta_shares)
        trade = {
            'delta_shares': modified_delta_shares,
            'avg_price': self.prev_price
        }
        self.position.transact_trade(trade)
        assert self.position.quantity <= self.max_quantity, f"delta_shares: {modified_delta_shares}, position: {self.position.__dict__}"
        
        # calculate reward
        delta_vt = self._delta_vt(modified_delta_shares)
        step_reward = self._calculate_reward(delta_vt)
        self.total_reward += step_reward
        
        self._update_portfolio(modified_delta_shares)
        
        # always update history last
        info = dict(
            actions = action.item(),
            delta_shares = modified_delta_shares,
            quantity = self.position.quantity,
            delta_vt = delta_vt,
            total_reward = self.total_reward,
            total_profit = self.total_profit,
            portfolio_value = self.portfolio_value,
            nav = self.nav,
            cash = self.cash,
            time = self.df['time'].iloc[self._current_tick],
            step_reward = step_reward,
            invested_value = self.invested_value,
            avg_price = self.position.avg_price,
        )
        self._update_history(info)
        return self._get_observation(), step_reward, self._is_done(), info
    
    def _preprocess(self, df):
        required_col = set('time open high low close volume'.split())
        check_col(df, required_col)
        df.sort_values(by='time', inplace=True)
        df = df.reset_index(drop=True)
        features = self.feature_extractor.preprocess(df)
        features = features.reset_index(drop=True)
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
        cash_percent = self.cash / self.portfolio_value
        stock_percent = self.nav / self.portfolio_value
        obs = np.append(features, cash_percent).astype(np.float32)
        obs = np.append(obs, stock_percent).astype(np.float32)
        return obs
    
    def _spread_cost(self, delta_shares: int) -> float:
        return abs(delta_shares) * self.tick_size

    def _impact_cost(self, delta_shares: int) -> float:
        return delta_shares ** 2 * self.tick_size / self.lot_size
    
    def _total_cost(self, delta_shares: int) -> float:
        return self._spread_cost(delta_shares) + self._impact_cost(delta_shares)
    
    def _delta_vt(self, delta_shares: int) -> float:
        return self.position.quantity * (self.price - self.prev_price) - self._total_cost(delta_shares)
    
    def _decode_action(self, action: float) -> int:
        target_cash = action * self.portfolio_value
        diff_value = self.cash - target_cash
        # if diff_value > 0 -> buy more shares
        # if diff_value < 0 -> sell shares
        diff_shares = int((diff_value / self.prev_price) / self.lot_size) * self.lot_size
        return diff_shares

    def _calculate_reward(self, delta_vt: float) -> float:
        # money_penalty = -self.cash * (1 - 0.04/365)
        return delta_vt #- 0.5 * self.kappa * (delta_vt ** 2) + money_penalty
    
    def get_history(self):
        history_df = pd.DataFrame(self.history)
        history_df = history_df.astype({'time':'datetime64[ns]'})
        data = self.df.merge(history_df, how='inner', on='time')
        return data
    
    def _is_done(self):
        done =  super()._is_done()
        returns = pd.Series(self.history['portfolio_value']).pct_change()
        maximum_drawdown = max_drawdown(returns)
        return done or (maximum_drawdown < -0.5)
    
    def _update_portfolio(self, delta_shares: int) -> float:
        # buy/sell at close price
        self.cash -= (delta_shares * self.prev_price + self._total_cost(delta_shares))
        assert self.cash >= 0
        
    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = 0, 1
        return low + (0.5 * (scaled_action + 1.0) * (high - low))