import unittest
from stock_env.envs import *
from stock_env.wrappers import StackAndSkipObs
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stock_env.data_loader import USTaskLoader


class TestRandomStockEnv(unittest.TestCase):
    def test_stack_and_skip_obs(self):
        data_loader = USTaskLoader.load("stock_env/datasets/sp500")
        env = TaskStockEnv(data_loader)
        task = env.sample_task()
        env.reset_task(task)

        env = StackAndSkipObs(env, 4, 4)
        # assert check_env(env)

        obs, info = env.reset()
        assert task == info["episode_ticker"]
        assert np.prod(
            env.unwrapped.observation_space.shape
        ) * env.num_stack == np.prod(obs.shape)
