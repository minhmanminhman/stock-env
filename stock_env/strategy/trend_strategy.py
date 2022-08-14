import pandas as pd
import pandas_ta as ta
import numpy as np

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

def trends(df: pd.DataFrame):
    # v2
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
        name=["closingmarubozu", "marubozu", "engulfing", "longline"], 
        scalar=1).sum(axis=1).astype(int)
    
    df['entry'] = ((entry_pattern > 0) * volume_breakout * (df['close'] > df['DCU_10_10'].shift())).astype(bool).astype(int)
    df['exit'] = (
        (exit_pattern < 0) 
        * volume_breakout 
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
