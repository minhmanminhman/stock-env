import finplot as fplt
import pandas as pd
from stock_env.data_loader.vndirect_loader import VNDDataLoader

vndloader = VNDDataLoader(symbols=["HCM"], start="2016-01-01", end="2022-12-31")
data = vndloader.download()

data = data.set_index("date")
fplt.candlestick_ochl(data[["adOpen", "adClose", "adHigh", "adLow"]])
fplt.show()
