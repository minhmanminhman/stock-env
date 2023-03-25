from stock_env.algos.ppo import PPO
from stock_env.algos.agent import Agent
from stock_env.envs import *
from stock_env.common.common_utils import open_config
from stock_env.common.env_utils import make_vec_env
import torch as th
import logging
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--startwith", "-s", type=str, default="A")

    args_ = parser.parse_args()
    startwith = args_.startwith

    env_id = "VNALL-v0"
    maml_model_path = "model/mamlpp_sp500_20230301_014749_all_time_best.pth"
    algo_config = "configs/ppo_adapt.yaml"

    args = open_config(algo_config, env_id=env_id)
    envs = make_vec_env(env_id, num_envs=args.num_envs, task=args.task)
    tickers = envs.envs[0].data_loader.tickers

    for ticker in tickers:
        if ticker.startswith(startwith):
            args = open_config(algo_config, env_id=env_id)
            # update task
            args.task = ticker
            args.run_name = f"adapt_{ticker}"

            logging.info(f"Training task: {ticker}")

            envs = make_vec_env(env_id, num_envs=args.num_envs, task=args.task)
            agent = Agent(envs, hiddens=args.hiddens)
            agent.load_state_dict(th.load(maml_model_path))
            algo = PPO(args=args, envs=envs, agent=agent)

            try:
                algo.learn()
            except KeyboardInterrupt:
                pass
            finally:
                algo.test()
                algo.close()
