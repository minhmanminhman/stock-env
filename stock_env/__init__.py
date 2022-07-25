from gym.envs.registration import register
import pandas as pd

register(
    id='SingleStock-v0',
    entry_point='stock_env.envs:SingleStockEnv',
    kwargs={
        'df': pd.read_csv('stock_env/datasets/SIMULATE_STOCK.csv', index_col=0),
        'max_trade_lot': 1,
        'max_lot': 1
    }
)

register(
    id='SimulatorStock-v0',
    entry_point='stock_env.envs:SimulatorStockEnv',
    kwargs={
        'size': 5000,
        'env_params': {'max_trade_lot': 1,'max_lot': 1}
    }
)