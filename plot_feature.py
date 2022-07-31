import finplot as fplt
import mt4_hst
import stock_env
from stock_env.utils import *
from stock_env.envs.vn_stock_env import VietnamStockEnv

def plot_format(df):
    df = df.reset_index(drop=True)
    df = df.astype({'time':'datetime64[ns]'})
    return df

df = pd.read_csv('temp/features.csv')
# plot format
df = plot_format(df)

symbol = 'FPT'
# create two axes
ax, ax2 = fplt.create_plot(symbol, rows=2)

# plot OHLCV
candles = df[['time','open','close','high','low']]
fplt.candlestick_ochl(candles, ax=ax)
volumes = df[['time','open','close','volume']]
fplt.volume_ocv(volumes, ax=ax.overlay())

# plot signals
breakout = df[(df['breakout']==1) & (df['volume_breakout']==1)]
breakout = plot_format(breakout)
fplt.plot(breakout['time'], breakout['low'] * 0.99, ax=ax, color='#408480', style='^', legend='Break out')



# restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()

# we're done
fplt.show()