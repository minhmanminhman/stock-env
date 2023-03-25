from stock_env.algos.ppo import PPO
from stock_env.algos.agent import Agent
from stock_env.envs import *
from stock_env.common.common_utils import open_config
from stock_env.common.env_utils import make_wrapped_env, make_vec_env
import logging
import argparse

if __name__ == "__main__":

    env_id = "MountainCarContinuous-v0"  # <<<<< CHANGE PARAM HERE
    algo_config = "configs/ppo.yaml"  # <<<<< CHANGE PARAM HERE

    args = open_config(algo_config, env_id=env_id)
    envs = make_vec_env(env_id, num_envs=args.num_envs)
    agent = Agent(envs, hiddens=args.hiddens)
    algo = PPO(args, envs, agent)
    try:
        algo.learn()
    except KeyboardInterrupt:
        pass
    finally:
        algo.test()
        algo.close()
