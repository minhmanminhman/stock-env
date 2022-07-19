from gym.envs.registration import register
from copy import deepcopy
import pandas as pd

register(
    id='SingleStock-v0',
    entry_point='stock_env.envs:SingleStockEnv',
    kwargs={
        'df': pd.read_csv('stock_env/datasets/data/SIMULATE_STOCK.csv', index_col=0),
    }
)

register(
    id='SimulatorStock-v0',
    entry_point='stock_env.envs:SimulatorStockEnv',
    kwargs={
        'size': 5000,
    }
)