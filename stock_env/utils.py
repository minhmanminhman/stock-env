import pandas as pd
import matplotlib.pyplot as plt
from pyfolio.plotting import (
    plot_monthly_returns_heatmap,
    plot_drawdown_underwater,
    plot_rolling_returns,
)
from empyrical import (
    annual_return,
    cum_returns_final,
    annual_volatility,
    sharpe_ratio,
    max_drawdown,
    value_at_risk
)

def plot_trade_log(data: pd.DataFrame):
    buy = data[data['delta_shares'] > 0]
    sell = data[data['delta_shares'] < 0]
    
    fig, ax = plt.subplots(2, 1, figsize=(15,8))
    
    ax[0].grid(True)
    ax[0].set_title('Trade log')
    #TODO: rename
    ax[0].plot(data.index, data['price'], label='Price')
    ax[0].scatter(buy.index, buy['price'], c='tab:green', marker='^', label='Long')
    ax[0].scatter(sell.index, sell['price'], c='tab:red', marker='v', label='Short')
    ax[0].legend()
    
    ax[1].grid(True)
    ax[1].set_title('Quantity')
    #TODO: rename
    ax[1].plot(data.index, data['quantity'], label='Quantity')
    ax[1].legend()
    
def plot_trade_log_v2(data: pd.DataFrame):
    buy = data[data['delta_shares'] > 0]
    sell = data[data['delta_shares'] < 0]
    
    fig, ax = plt.subplots(2, 1, figsize=(15,8))
    
    ax[0].grid(True)
    ax[0].set_title('Trade log')
    #TODO: rename
    ax[0].plot(data['time'], data['close'], label='Price')
    ax[0].scatter(buy['time'], buy['close'], c='tab:green', marker='^', label='Long')
    ax[0].scatter(sell['time'], sell['close'], c='tab:red', marker='v', label='Short')
    ax[0].legend()
    
    ax[1].grid(True)
    ax[1].set_title('Quantity')
    #TODO: rename
    ax[1].plot(data.index, data['quantity'], label='Quantity')
    ax[1].legend()

def create_performance(returns: pd.Series):
    print(
    f"""
    Annual return     : {annual_return(returns) * 100: .2f}%
    Cumulative return : {cum_returns_final(returns) * 100: .2f}%
    Sharpe ratio      : {sharpe_ratio(returns): .2f}
    Maximum Drawdown  : {max_drawdown(returns) * 100: .2f}%
    Annual Volatility : {annual_volatility(returns) * 100: .2f}%
    Value-At-Risk     : {value_at_risk(returns) * 100: .2f}%
    """)
    
    fig, axs = plt.subplots(3,1, figsize=(12,10))
    plot_rolling_returns(returns, ax=axs[0])
    plot_drawdown_underwater(returns, ax=axs[1])
    plot_monthly_returns_heatmap(returns, ax=axs[2])
    
def check_col(x: pd.DataFrame, to_check):
   if not set(to_check).issubset(set(x.columns)):
      msg = f"{' and '.join(set(to_check).difference(x.columns))} are missing in the dataframe"
      assert False, msg

def plot_format(df):
    df = df.reset_index(drop=True)
    df = df.astype({'time':'datetime64[ns]'})
    return df

def plot_signals(df, ax):
    """
    Plot signals generating from ta.tsignals
    """
    # check columns
    required_col = set('time open high low close volume TS_Entries TS_Exits'.split())
    check_col(df, required_col)
    import finplot as fplt
    
    # format dataset
    df = plot_format(df)
    
    # plot OHLCV
    candles = df[['time','open','close','high','low']].copy()
    fplt.candlestick_ochl(candles, ax=ax)
    volumes = df[['time','open','close','volume']].copy()
    fplt.volume_ocv(volumes, ax=ax.overlay())

    # plot signals
    try:
        buy = df[['time', 'TS_Entries', 'low']].copy()
        buy['TS_Entries'] = buy['TS_Entries'].astype(bool)
        buy = buy[buy['TS_Entries'] == True]
        buy = plot_format(buy)
        assert len(buy) > 0
    except:
        pass
    else:
        fplt.plot(buy['time'], buy['low'] * 0.99, ax=ax, color='#408480', style='^', legend='Long')

    try:
        sell = df[['time', 'TS_Exits', 'high']].copy()
        sell['TS_Exits'] = sell['TS_Exits'].astype(bool)
        sell = sell[sell['TS_Exits'] == True]
        sell = plot_format(sell)
        assert len(sell) > 0
    except:
        pass
    else:
        fplt.plot(sell['time'], sell['high'] * 1.01, ax=ax, color='#ee0e00', style='v', legend='Short')