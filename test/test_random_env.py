import os
import sys
sys.path.append(os.path.realpath('./'))
import unittest
import pytest
from stable_baselines3.common.env_checker import check_env

from stock_env.data_loader import RandomStockLoader
from stock_env.envs import RandomStockEnv
from stock_env.feature import TrendFeatures
from stock_env.wrappers import StackObs

class TestRandomStockEnv(unittest.TestCase):
    
    def test_random_stock_env(self):
        data_loader = RandomStockLoader(
            tickers = "SSI HPG VNM".split(),
            data_folder_path = "../stock_datasets/",
            feature_extractor = TrendFeatures
        )

        env = RandomStockEnv(data_loader)
        check_env(env)
    
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_buffer_wrapper(self):
        data_loader = RandomStockLoader(
            tickers = "SSI HPG VNM".split(),
            data_folder_path = "../stock_datasets/",
            feature_extractor = TrendFeatures
        )

        env = RandomStockEnv(data_loader)
        env = StackObs(env, 5)
        check_env(env)