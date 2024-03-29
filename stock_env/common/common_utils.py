import warnings

warnings.filterwarnings("ignore")
import yaml
from types import SimpleNamespace
from typing import List, Dict, Tuple, Union, Callable, Any
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
    value_at_risk,
    roll_max_drawdown
)


def plot_trade_log(data: pd.DataFrame):
    buy = data[data["delta_shares"] > 0]
    sell = data[data["delta_shares"] < 0]

    fig, ax = plt.subplots(2, 1, figsize=(15, 8))

    ax[0].grid(True)
    ax[0].set_title("Trade log")
    # TODO: rename
    ax[0].plot(data.index, data["price"], label="Price")
    ax[0].scatter(buy.index, buy["price"], c="tab:green", marker="^", label="Long")
    ax[0].scatter(sell.index, sell["price"], c="tab:red", marker="v", label="Short")
    ax[0].legend()

    ax[1].grid(True)
    ax[1].set_title("Quantity")
    # TODO: rename
    ax[1].plot(data.index, data["quantity"], label="Quantity")
    ax[1].legend()


def plot_trade_log_v2(data: pd.DataFrame):
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    ticker = data["ticker_x"].unique()[0]
    ax[0].set_title(f"Trade log, Ticker: {ticker}")
    # TODO: rename
    ax[0].plot(data["time"], data["close"], label="Price")

    try:
        buy = data[data["delta_shares"] > 0]
        assert len(buy) > 0
        ax[0].scatter(
            buy["time"], buy["close"], c="tab:green", marker="^", label="Long"
        )
    except:
        pass

    try:
        sell = data[data["delta_shares"] < 0]
        assert len(sell) > 0
        ax[0].scatter(
            sell["time"], sell["close"], c="tab:red", marker="v", label="Short"
        )
    except:
        pass

    ax[0].legend()

    ax[1].grid(True)
    ax[1].set_title("% Cash")
    # TODO: rename
    ax[1].plot(data["time"], data["cash"] / data["portfolio_value"], label="% Cash")
    ax[1].legend()


def create_performance(returns: pd.Series, plot=True):
    # apply many functions to returns
    l_func = [
        annual_return,
        cum_returns_final,
        sharpe_ratio,
        max_drawdown,
        annual_volatility,
        value_at_risk,
    ]
    l_results = map(lambda x: x(returns), l_func)

    results = {}
    for func, result in zip(l_func, l_results):
        results.update({str(func.__name__): result})

    print(
        f"""
    Annual return     : {results['annual_return'] * 100: .2f}%
    Cumulative return : {results['cum_returns_final'] * 100: .2f}%
    Sharpe ratio      : {results['sharpe_ratio']: .2f}
    Maximum Drawdown  : {results['max_drawdown'] * 100: .2f}%
    Annual Volatility : {results['annual_volatility'] * 100: .2f}%
    Value-At-Risk     : {results['value_at_risk'] * 100: .2f}%
    """
    )
    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("Trading performance")
        plot_rolling_returns(returns, ax=axs[0])
        plot_drawdown_underwater(returns, ax=axs[1])
        plot_monthly_returns_heatmap(returns, ax=axs[2])

    return results


def check_col(x: pd.DataFrame, to_check):
    if not set(to_check).issubset(set(x.columns)):
        msg = f"{' and '.join(set(to_check).difference(x.columns))} are missing in the dataframe"
        assert False, msg


def plot_format(df):
    df = df.reset_index(drop=True)
    df = df.astype({"time": "datetime64[ns]"})
    return df


def plot_signals(df, ax):
    """
    Plot signals generating from ta.tsignals
    """
    # check columns
    required_col = set("time open high low close volume".split())
    check_col(df, required_col)
    import finplot as fplt

    # format dataset
    df = plot_format(df)

    # plot OHLCV
    candles = df[["time", "open", "close", "high", "low"]].copy()
    fplt.candlestick_ochl(candles, ax=ax)
    volumes = df[["time", "open", "close", "volume"]].copy()
    fplt.volume_ocv(volumes, ax=ax.overlay())

    # plot signals
    try:
        buy = df[["time", "TS_Entries", "low"]].copy()
        buy["TS_Entries"] = buy["TS_Entries"].astype(bool)
        buy = buy[buy["TS_Entries"] == True]
        buy = plot_format(buy)
        assert len(buy) > 0
    except:
        pass
    else:
        fplt.plot(
            buy["time"],
            buy["low"] * 0.99,
            ax=ax,
            color="#408480",
            style="^",
            legend="Long",
            width=10,
        )

    try:
        sell = df[["time", "TS_Exits", "high"]].copy()
        sell["TS_Exits"] = sell["TS_Exits"].astype(bool)
        sell = sell[sell["TS_Exits"] == True]
        sell = plot_format(sell)
        assert len(sell) > 0
    except:
        pass
    else:
        fplt.plot(
            sell["time"],
            sell["high"] * 1.01,
            ax=ax,
            color="#ee0e00",
            style="v",
            legend="Short",
            width=10,
        )


def plot_signals_from_history(df, ax):
    """
    Plot signals generating from ta.tsignals
    """
    # check columns
    required_col = set("time open high low close volume".split())
    check_col(df, required_col)
    import finplot as fplt

    # format dataset
    df = plot_format(df)

    # plot OHLCV
    candles = df[["time", "open", "close", "high", "low"]].copy()
    fplt.candlestick_ochl(candles, ax=ax)
    volumes = df[["time", "open", "close", "volume"]].copy()
    fplt.volume_ocv(volumes, ax=ax.overlay())

    # plot signals
    try:
        buy = df[df["delta_shares"] > 0]
        assert len(buy) > 0
        buy = plot_format(buy)
        assert len(buy) > 0
    except:
        pass
    else:
        fplt.plot(
            buy["time"],
            buy["low"] * 0.99,
            ax=ax,
            color="#408480",
            style="^",
            legend="Long",
            width=3,
        )

    try:
        sell = df[df["delta_shares"] < 0]
        assert len(sell) > 0
        sell = plot_format(sell)
        assert len(sell) > 0
    except:
        pass
    else:
        fplt.plot(
            sell["time"],
            sell["high"] * 1.01,
            ax=ax,
            color="#ee0e00",
            style="v",
            legend="Short",
            width=3,
        )


def plot_quantity(df, ax):
    """
    Plot signals generating from ta.tsignals
    """
    # check columns
    required_col = set("time quantity".split())
    check_col(df, required_col)
    import finplot as fplt

    # format dataset
    df = plot_format(df)

    # plot quantity
    df["quantity"].plot(ax=ax, legend="Quantity")


def open_config(path, env_id, is_args=True) -> Union[SimpleNamespace, Dict]:

    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    try:
        args = config[env_id]
        if is_args:
            args = SimpleNamespace(**args)
    except KeyError:
        raise KeyError(f"Environment {env_id} not found in config file")

    return args


def get_linear_fn(start: float, end: float, end_fraction: float) -> Callable:
    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def complement_action(action, action_space):
    low = action_space.low
    high = action_space.high
    return high - (action - low)
