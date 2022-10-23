from .base_env import *
from .random_stock import *

def create_env(file):
    def make_env():
        from ..data_loader import RandomStockLoader
        from ..wrappers import StackObs
        import pathlib
        path = pathlib.Path(__file__).parent.parent.joinpath('datasets').resolve()
        data_loader = RandomStockLoader.load(f"{path}/{file}")
        env = RandomStockEnv(data_loader)
        env = StackObs(env, n_steps=5)
        return env
    return make_env

register(
    id=f"FinService-v0",
    entry_point=create_env('finservice_data_loader'),
)

register(
    id=f"RandomVN30-v0",
    entry_point=create_env('30stocks'),
)

register(
    id=f"SP500-v0",
    entry_point=create_env('sp500_data_loader'),
)