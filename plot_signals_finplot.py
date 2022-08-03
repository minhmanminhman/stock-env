import finplot as fplt
from stock_env.utils import plot_signals
import pandas as pd
import pandas_ta as ta

symbol = 'FPT'
df = pd.read_csv('temp/df_signals.csv')
df.ta.sma(50, append=True)
df.ta.ema(10, append=True)
df.ta.ema(20, append=True)
df.ta.donchian(df['high'], df['close'], lower_length=20, upper_length=20, append=True)
df.ta.donchian(df['high'], df['close'], lower_length=20, upper_length=20, append=True)

ax = fplt.create_plot(symbol)

plot_signals(df, ax)

# restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()
# we're done
fplt.show()