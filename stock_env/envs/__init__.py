from .base_env import *
from .random_stock import *
from stock_env.envs.task_stock import *
from stock_env.envs.vec_task_env import *
from stock_env.wrappers import StackAndSkipObs
from stock_env.common.common_utils import open_config
from gymnasium import register
import pathlib

config_path = pathlib.Path(__file__).parent.parent.parent.joinpath("configs").resolve()


def create_env(name):
    def make_env():
        from ..data_loader import RandomStockLoader
        from ..wrappers import StackObs
        import pathlib

        path = pathlib.Path(__file__).parent.parent.joinpath("datasets").resolve()
        data_loader = RandomStockLoader.load(f"{path}/{name}")
        env = RandomStockEnv(data_loader)
        env = StackObs(env, n_steps=5)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return make_env


def make_task_env(name, gamma=0.99, env_kwargs={}):
    def _thunk():
        from stock_env.data_loader import USTaskLoader
        import pathlib

        path = pathlib.Path(__file__).parent.parent.joinpath("datasets").resolve()

        task_loader = USTaskLoader.load(f"{path}/{name}")
        env = TaskStockEnv(task_loader, **env_kwargs)

        # temp reset_task to create env, should be reset later
        _task = env.sample_task()
        env.reset_task(_task)

        # wrap env
        # env = StackAndSkipObs(env, num_stack=5, num_skip=3)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return _thunk


def make_env(name, seed=None, gamma=0.99):
    def _thunk():
        from stock_env.data_loader import USTaskLoader
        import pathlib

        path = pathlib.Path(__file__).parent.parent.joinpath("datasets").resolve()

        task_loader = USTaskLoader.load(f"{path}/{name}")
        env = TaskStockEnv(task_loader)

        _task = env.sample_task(seed=seed)
        env.reset_task(_task)

        # wrap env
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        return env

    return _thunk


register(
    id=f"FinService-v0",
    entry_point=create_env("finservice_data_loader"),
)

register(
    id=f"RandomVN30-v0",
    entry_point=create_env("30stocks"),
)

register(
    id=f"SP500-v0",
    entry_point=make_task_env(
        name="sp500",
        env_kwargs=open_config(
            f"{config_path}/envs.yaml", env_id="SP500-v0", is_args=False
        ),
    ),
)

register(
    id=f"VNALL-v0",
    entry_point=make_task_env(
        name="vnall",
        env_kwargs=open_config(
            f"{config_path}/envs.yaml", env_id="VNALL-v0", is_args=False
        ),
    ),
)

register(
    id=f"FAANGTask-v0",
    entry_point=make_task_env("faang_task_loader"),
)

register(
    id=f"MiniFAANG-v0",
    entry_point=make_task_env(
        name="mini_faang",
        env_kwargs=open_config(
            f"{config_path}/envs.yaml", env_id="MiniFAANG-v0", is_args=False
        ),
    ),
)

register(
    id="MiniVNStock-v0",
    entry_point=make_task_env(
        name="mini_vnstock",
        env_kwargs=open_config(
            f"{config_path}/envs.yaml", env_id="MiniVNStock-v0", is_args=False
        ),
    ),
)
