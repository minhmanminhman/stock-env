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