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

def plot_trade_log(data: pd.DataFrame, ax):
    buy = data[data['delta_shares'] > 0]
    sell = data[data['delta_shares'] < 0]

    ax.grid(True)
    ax.set_title('Trade log')
    #TODO: rename
    ax.plot(data.index, data['A'], label='Price')
    ax.scatter(buy.index, buy['A'], c='tab:green', marker='^', label='Long')
    ax.scatter(sell.index, sell['A'], c='tab:red', marker='v', label='Short')
    ax.legend()

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