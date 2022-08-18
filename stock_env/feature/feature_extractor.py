from abc import abstractmethod
import pandas as pd
import pandas_ta as ta
from ..utils import check_col

class BaseFeaturesExtractor:
    
    @abstractmethod
    def preprocess(self):
        raise NotImplementedError

class OneStockFeatures(BaseFeaturesExtractor):
    """
    Indicators in paper: 
    Deep Reinforcement Learning approach using customized technical indicators 
    for a pre-emerging market: A case study of Vietnamese stock market
    """
    
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
        self.feature_cols = "close ADX_14 SMA_10 RSI_14 CCI_14_0.015 BBL_5_2.0 BBU_5_2.0 MACD_12_26_9".split()
        self.feature_dim = len(self.feature_cols)
    
    def preprocess(self, df):
        check_col(df, self.required_cols)
        df.sort_values(by='time', inplace=True)
        # create indicators
        df.ta.strategy(self.strategy)
        
        df.dropna(inplace=True)
        return df[self.feature_cols]