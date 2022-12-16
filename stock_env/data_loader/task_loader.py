import numpy as np
from stock_env.feature.feature_extractor import BaseFeaturesExtractor, TrendFeatures
from stock_env.data_loader import USStockLoader, BaseTaskLoader, VNStockLoader

class USTaskLoader(USStockLoader, BaseTaskLoader):
    
    def __init__(
        self, 
        tickers: list,
        feature_extractor: BaseFeaturesExtractor = TrendFeatures,
        max_episode_steps: int = 250,
        test_pct: float = 0.2,
        data_period: str = '1y'
    ):
        super().__init__(
            tickers=tickers,
            feature_extractor=feature_extractor,
            max_episode_steps=max_episode_steps,
            data_period=data_period
        )
        self.episode_ticker = None
        self.test_pct = test_pct
    
    def reset_task(self, task: str) -> None:
        """Each task is a different ticker"""
        assert task in self.tickers, f"{task} not in available tickers"
        self.episode_ticker = str(task)
    
    def sample_task(self) -> str:
        return np.random.choice(self.tickers, size=1).item()
    
    def reset(self):
        if self.episode_ticker is None:
            raise RuntimeError(
                'Should call `reset_task` before calling `reset` '
                'to get task. Call `sample_task` to get available tickers')
        self.ohlcv = self.stack_ohlcv.loc[self.episode_ticker]
        self.features = self.stack_features.loc[self.episode_ticker]
        self._end_tick = self.ohlcv.shape[0] - 1
        start_idxes = np.arange(start=0, stop=self.ohlcv.shape[0], step=self.max_episode_steps)
        test_mask = (start_idxes >= np.quantile(start_idxes, 1 - self.test_pct))
        self.train_idxes = start_idxes[~test_mask]
        self.test_idxes = start_idxes[test_mask]
        
        if self.train_mode:
            idxes = self.train_idxes
        else:
            idxes = self.test_idxes
        assert len(idxes) > 0
        start_tick = np.random.choice(idxes)
        end_tick = min(start_tick + self.max_episode_steps, self._end_tick)
        
        self._current_tick = self._start_tick = start_tick
        self._end_episode_tick = end_tick
        
        return self._step(), self._reset_info()

class VNTaskLoader(VNStockLoader, BaseTaskLoader):
    
    def __init__(
        self, 
        tickers: list,
        feature_extractor: BaseFeaturesExtractor,
        max_episode_steps: int = 250,
        start_date: str = '2016-01-01',
        end_date: str = '2022-12-31',
        test_pct: float = 0.2,
    ):
        super().__init__(
            tickers=tickers,
            feature_extractor=feature_extractor,
            max_episode_steps=max_episode_steps,
            start_date=start_date,
            end_date=end_date
        )
        self.episode_ticker = None
        self.test_pct = test_pct
    
    def reset_task(self, task: str) -> None:
        """Each task is a different ticker"""
        assert task in self.tickers, f"{task} not in available tickers"
        self.episode_ticker = str(task)
    
    def sample_task(self) -> str:
        return np.random.choice(self.tickers, size=1).item()
    
    def reset(self):
        if self.episode_ticker is None:
            raise RuntimeError(
                'Should call `reset_task` before calling `reset` '
                'to get task. Call `sample_task` to get available tickers')
        self.ohlcv = self.stack_ohlcv.loc[self.episode_ticker]
        self.features = self.stack_features.loc[self.episode_ticker]
        self._end_tick = self.ohlcv.shape[0] - 1
        start_idxes = np.arange(start=0, stop=self.ohlcv.shape[0], step=self.max_episode_steps)
        test_mask = (start_idxes >= np.quantile(start_idxes, 1 - self.test_pct))
        self.train_idxes = start_idxes[~test_mask]
        self.test_idxes = start_idxes[test_mask]
        
        if self.train_mode:
            idxes = self.train_idxes
        else:
            idxes = self.test_idxes
        
        assert len(idxes) > 0
        start_tick = np.random.choice(idxes)
        end_tick = min(start_tick + self.max_episode_steps, self._end_tick)
        
        self._current_tick = self._start_tick = start_tick
        self._end_episode_tick = end_tick
        
        return self._step(), self._reset_info()