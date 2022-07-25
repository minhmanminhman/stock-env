import finplot as fplt
import numpy as np
import mt4_hst
import pandas as pd
import pandas_ta as ta

####
import finplot as fplt
import yfinance

df = yfinance.download('AAPL')
df[['Open', 'Close', 'High', 'Low']]
print(df.head())
###

# # pull some data
# symbol = 'FPT'

# # format it in pandas
# df = mt4_hst.read_hst("stock_env/datasets/FPT1440.hst")
# df.ta.rsi(append=True)
# df.dropna(inplace=True)

# # create two axes
# ax, ax2 = fplt.create_plot(symbol, rows=2)

# # plot candle sticks
# # candles = df[['time','open','close','high','low']]
# # fplt.candlestick_ochl(candles, ax=ax)
# df.plot(kind='candle', ax=ax)

# # overlay volume on the top plot
# # volumes = df[['time','open','close','volume']]
# # fplt.volume_ocv(volumes, ax=ax.overlay())

# # draw some random crap on our second plot
# # fplt.plot(df['time'], df['RSI_14'], ax=ax2, legend='RSI_14')
# # fplt.set_y_range(-1.4, +3.7, ax=ax2) # hard-code y-axis range limitation

# # restore view (X-position and zoom) if we ever run this example again
# fplt.autoviewrestore()

# # we're done
# fplt.show()