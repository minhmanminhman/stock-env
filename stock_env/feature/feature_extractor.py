from abc import abstractmethod
import pandas as pd
import pandas_ta as ta
from ..utils import check_col
import numpy as np
from ..strategy.trend_strategy import *

class BaseFeaturesExtractor:
    
    @abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

class OneStockFeatures(BaseFeaturesExtractor):
    """
    Indicators in paper: 
    Deep Reinforcement Learning approach using customized technical indicators 
    for a pre-emerging market: A case study of Vietnamese stock market
    """
    
    def __init__(self):
        self.strategy = CommonStrategy
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

class TrendFeatures(BaseFeaturesExtractor):
    """
    Indicators in paper: 
    Deep Reinforcement Learning approach using customized technical indicators 
    for a pre-emerging market: A case study of Vietnamese stock market
    """
    
    def __init__(self):
        self.strategy = TrendStrategy
        self.required_cols = set('time open high low close volume'.split())
        self.feature_cols = \
            """
            TS_Trends ADX_20 AROOND_20 AROONU_20 
            AROONOSC_20 STC_10_10_20_0.5 STCstoch_10_10_20_0.5 
            NATR_20 RSI_20 CCI_20_0.015
            """.split()
        self.feature_dim = len(self.feature_cols)
    
    def _create_trend(self, df: pd.DataFrame):
        trend_cond = (
            (df['close'] > df['SMA_50']).astype(int)
            + (df['EMA_10'] > df['EMA_20']).astype(int)
            + (df['DCL_10_10'] > df['DCL_50_50']).astype(int)
        ) >= 3
        return trend_cond
    
    def _modify_entry_exit(self, df: pd.DataFrame):
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
    
    def _breakout_entry(self, df: pd.DataFrame):
        volume_breakout = (df['volume'] >= 1.25 * df['VOLUME_SMA_20']).astype(int)
    
        # candle pattern
        pattern = df.ta.cdl_pattern(
            name=["closingmarubozu", "marubozu", "engulfing", "longline"], 
            scalar=1).sum(axis=1).astype(int)
        
        df['entry'] = ((pattern > 0) * volume_breakout * (df['close'] > df['DCU_10_10'].shift())).astype(bool).astype(int)
        df['exit'] = ((pattern < 0) * volume_breakout ).astype(bool).astype(int)
        df['original_trends'] = df['trends']
        df['trends'] = df.groupby((df.trends != df.trends.shift()).cumsum()).apply(self._modify_entry_exit).to_list()
        return df
    
    def preprocess(self, df, asbool=False, return_all=False):
        check_col(df, self.required_cols)
        df.sort_values(by='time', inplace=True)
        df.ta.strategy(self.strategy)
        df['trends'] = self._create_trend(df)
        df = self._breakout_entry(df)
        df.ta.tsignals(df['trends'], asbool=asbool, append=True)
        df['CCI_20_0.015'] = df['CCI_20_0.015'] / 100
        df[['STC_10_10_20_0.5', 'STCstoch_10_10_20_0.5']] = df[['STC_10_10_20_0.5', 'STCstoch_10_10_20_0.5']] / 100
        df.dropna(inplace=True)
        
        if return_all:
            return df
        else:
            return df[self.feature_cols]