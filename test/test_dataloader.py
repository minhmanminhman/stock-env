import unittest
from stock_env.envs import *
from stock_env.data_loader import USTaskLoader
from stock_env.feature.feature_extractor import *
from stock_env.data_loader import *


class TestRandomStockEnv(unittest.TestCase):
    def test_reset_sp500_data_loader(self):
        sp500_loader = USTaskLoader.load(f"stock_env/datasets/sp500")
        for ticker in sp500_loader.tickers:
            sp500_loader.reset_task(ticker)
            try:
                obs = sp500_loader.reset()
            except:
                print(f"Error: {ticker}")
                print(sp500_loader.ohlcv.shape)
                continue

    def test_reset_vnall_data_loader(self):
        vnall = VNTaskLoader.load(f"stock_env/datasets/vnall")
        for ticker in vnall.tickers:
            vnall.reset_task(ticker)
            try:
                obs = vnall.reset()
            except:
                print(f"Error: {ticker}")
                print(vnall.ohlcv.shape)
                continue
