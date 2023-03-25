from stock_env.algos.ppo import PPO
from stock_env.algos.agent import Agent
from stock_env.envs import *
from stock_env.common.common_utils import open_config
from stock_env.common.env_utils import make_wrapped_env, make_vec_env
import logging
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--startwith", "-s", type=str, default="A")

    args_ = parser.parse_args()
    startwith = args_.startwith

    env_id = "VNALL-v0"  # <<<<< CHANGE PARAM HERE
    algo_config = "configs/ppo_cp2.yaml"  # <<<<< CHANGE PARAM HERE

    args = open_config(algo_config, env_id=env_id)
    envs = make_vec_env(env_id, num_envs=args.num_envs, task=args.task)
    tickers = envs.envs[0].data_loader.tickers

    for ticker in tickers:
        if ticker.startswith(startwith):
            args = open_config(algo_config, env_id=env_id)
            # update task
            args.task = ticker
            # args.run_name = ticker
            args.run_name = f"explore0.2_{ticker}"

            logging.info(f"Training task: {ticker}")

            envs = make_vec_env(env_id, num_envs=args.num_envs, task=args.task)
            agent = Agent(envs, hiddens=args.hiddens)
            algo = PPO(args, envs, agent)
            try:
                algo.learn()
            except KeyboardInterrupt:
                pass
            finally:
                algo.test()
                algo.close()
