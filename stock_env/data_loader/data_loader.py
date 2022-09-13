import pandas as pd
import mt4_hst
import numpy as np

from .base import BaseDataLoader
from ..feature import BaseFeaturesExtractor
from ..utils import check_col

class RandomStockLoader(BaseDataLoader):
    
    def __init__(
        self, 
        tickers: list,
        data_folder_path: str,
        feature_extractor: BaseFeaturesExtractor,
        max_episode_steps: int = 250,
    ):
        self.tickers = tickers
        self.data_folder_path = data_folder_path
        self.max_episode_steps = max_episode_steps
        self.train_mode = True
        
        self.feature_extractor = feature_extractor()
        df = self._load_data()
        self.stack_features, self.stack_ohlcv = self._preprocess(df)
        # episode
        self._start_tick = None
        self._end_tick = None
        self._end_episode_tick = None
        self._current_tick = None
    
    def _load_data(self):
        df = pd.DataFrame()
        required_col = set('time open high low close volume'.split())
        for ticker in self.tickers:
            _df = mt4_hst.read_hst(self.data_folder_path + ticker + "1440.hst")
            check_col(_df, required_col)
            _df.sort_values(by='time', inplace=True)
            _df['ticker'] = ticker
            df = pd.concat([df, _df])
        return df
    
    def _preprocess(self, df):
        grouped_ohlcv = df.groupby('ticker')
        stack_features = grouped_ohlcv.apply(lambda x: self.feature_extractor.preprocess(x))
        
        stack_ohlcv = df.set_index(keys=['ticker', df.index])
        stack_ohlcv, _ = stack_ohlcv.align(stack_features, join='inner', axis=0)
        return stack_features, stack_ohlcv
    
    @property
    def current_ohlcv(self):
        """Get current OHLCV"""
        if not hasattr(self, 'ohlcv'):
            raise ValueError("Call reset() method to have attribute 'ohlcv'")
        return self.ohlcv.iloc[self._current_tick]
    
    @property
    def ticker(self):
        """Get current OHLCV"""
        if not hasattr(self, 'episode_ticker'):
            raise ValueError("Call reset() method to have attribute 'episode_ticker'")
        return self.episode_ticker
    
    @property
    def is_done(self):
        return self._current_tick == self._end_episode_tick
    
    @property
    def feature_dim(self):
        return self.feature_extractor.feature_dim
    
    def reset(self):
        # random choose stock and start period
        self.episode_ticker = np.random.choice(self.tickers, size=1).item()
        self.ohlcv = self.stack_ohlcv.loc[self.episode_ticker]
        self.features = self.stack_features.loc[self.episode_ticker]
        self._end_tick = self.ohlcv.shape[0] - 1
        
        if not self.train_mode:
            start_tick = 0
            end_tick = self._end_tick
        else:
            start_idxes = np.arange(start=0, stop=self.ohlcv.shape[0], step=self.max_episode_steps)
            start_tick = np.random.choice(start_idxes)
            end_tick = min(start_tick + self.max_episode_steps, self._end_tick)
        
        self._current_tick = self._start_tick = start_tick
        self._end_episode_tick = end_tick
        
        return self._step()
    
    def step(self):
        self._current_tick += 1
        return self._step()
    
    def _step(self):
        return self.features.iloc[self._current_tick].values
    
    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.train_mode = mode
        return self