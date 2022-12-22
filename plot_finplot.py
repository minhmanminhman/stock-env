import finplot as fplt
import mt4_hst
import stock_env
from stock_env.utils import *
from stock_env.envs.vn_stock_env import VietnamStockEnv


def plot_format(df):
    df = df.reset_index(drop=True)
    df = df.astype({"time": "datetime64[ns]"})
    return df


df = pd.read_csv("temp/history.csv")
# plot format
df = plot_format(df)

symbol = "FPT"
# create two axes
ax, ax2 = fplt.create_plot(symbol, rows=2)

# plot OHLCV
candles = df[["time", "open", "close", "high", "low"]]
fplt.candlestick_ochl(candles, ax=ax)
volumes = df[["time", "open", "close", "volume"]]
fplt.volume_ocv(volumes, ax=ax.overlay())

# draw RSI
fplt.set_y_range(0, 100, ax=ax2)
df["RSI_20"].plot(ax=ax2, legend="RSI")
fplt.add_band(30, 70, ax=ax2)

# plot signals
try:
    buy = df[df["delta_shares"] > 0]
    buy = plot_format(buy)
    assert len(buy) > 0
except:
    pass
else:
    fplt.plot(
        buy["time"], buy["low"] * 0.99, ax=ax, color="#408480", style="^", legend="Long"
    )

try:
    sell = df[df["delta_shares"] < 0]
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
    )

# restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()

# we're done
fplt.show()
