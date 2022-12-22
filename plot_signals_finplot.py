import finplot as fplt
from stock_env.utils import *
import pandas as pd
import pandas_ta as ta
from stock_env.feature.feature_extractor import *

env = "VietnamStockContinuousEnv"
algo = "ppo"
ticker = "SSI"
path = "../stock_datasets/"
feature_extractor = TrendFeatures()
# name = f"ticker_history_{algo}_{env}_{feature_extractor.__class__.__name__}"
name = f"PPO_BufferWrapper_TrendFeatures_finservice"

df = pd.read_csv(f"temp/history/{name}.csv")
df.sort_values(by="time", inplace=True)
df.index = pd.to_datetime(df["time"])
df = plot_format(df)

ax = fplt.create_plot(ticker, rows=2)

# plot_signals(df, ax)
plot_signals_from_history(df, ax[0])
plot_quantity(df, ax[1])
# df['SMA_50'].plot(ax=ax[0], legend='MA50')
# df['EMA_10'].plot(ax=ax[0], legend='EMA10')
# df['EMA_20'].plot(ax=ax[0], legend='EMA20')
# df['DCL_10_10'].plot(ax=ax[0], legend='LOW_10')
# df['DCL_20_20'].plot(ax=ax[0], legend='LOW_20')
# df['DCL_50_50'].plot(ax=ax[0], legend='LOW_50')

# try:
#     cdl_pattern = df[['time', 'candle_pattern', 'high']].copy()
#     cdl_pattern = cdl_pattern[cdl_pattern['candle_pattern'] == True]
#     cdl_pattern = plot_format(cdl_pattern)
#     assert len(cdl_pattern) > 0
# except:
#     pass
# else:
#     fplt.plot(cdl_pattern['time'], cdl_pattern['high'] * 1.01, ax=ax[0], color='#ee0e00', style='crosshair', legend='candle_pattern')

# restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()
# we're done
fplt.show()
