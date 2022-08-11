import finplot as fplt
from stock_env.utils import plot_signals, plot_format
import pandas as pd
import pandas_ta as ta

symbol = 'MBB'
df = pd.read_csv('temp/signal/' + symbol + '.csv')
df = df[df['time'] >= '2021-01-01']
df = plot_format(df)

ax = fplt.create_plot(symbol)

plot_signals(df, ax)
df['SMA_50'].plot(ax=ax, legend='MA50')
df['EMA_10'].plot(ax=ax, legend='EMA10')
df['EMA_20'].plot(ax=ax, legend='EMA20')
df['DCL_10_10'].plot(ax=ax, legend='LOW_10')
df['DCL_20_20'].plot(ax=ax, legend='LOW_20')
df['DCL_50_50'].plot(ax=ax, legend='LOW_50')

try:
    cdl_pattern = df[['time', 'candle_pattern', 'high']].copy()
    cdl_pattern = cdl_pattern[cdl_pattern['candle_pattern'] == True]
    cdl_pattern = plot_format(cdl_pattern)
    assert len(cdl_pattern) > 0
except:
    pass
else:
    fplt.plot(cdl_pattern['time'], cdl_pattern['high'] * 1.01, ax=ax, color='#ee0e00', style='crosshair', legend='candle_pattern')

# restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()
# we're done
fplt.show()