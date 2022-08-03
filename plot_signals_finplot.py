import finplot as fplt
from stock_env.utils import plot_signals, plot_format
import pandas as pd
import pandas_ta as ta

symbol = 'FPT'
df = pd.read_csv('temp/df_signals.csv')
df = df[df['time'] >= '2018-01-01']
df = plot_format(df)

ax = fplt.create_plot(symbol)

plot_signals(df, ax)
df['SMA_50'].plot(ax=ax, legend='MA50')
df['EMA_10'].plot(ax=ax, legend='EMA10')
df['EMA_20'].plot(ax=ax, legend='EMA20')
df['DCL_20_20'].plot(ax=ax, legend='LOW_20')
df['DCL_50_50'].plot(ax=ax, legend='LOW_50')

try:
    cdl_pattern = df[['time', 'CDL_MARUBOZU', 'high']].copy()
    cdl_pattern = cdl_pattern[cdl_pattern['CDL_MARUBOZU'] != 0]
    cdl_pattern = plot_format(cdl_pattern)
    assert len(cdl_pattern) > 0
except:
    pass
else:
    fplt.plot(cdl_pattern['time'], cdl_pattern['high'] * 1.01, ax=ax, color='#ee0e00', style='crosshair', legend='maru')

# restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()
# we're done
fplt.show()