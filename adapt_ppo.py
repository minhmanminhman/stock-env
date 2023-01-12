from stock_env.algos.ppo import PPO
from stock_env.algos.agent import Agent
from stock_env.envs import *
from stock_env.common.common_utils import open_config
from stock_env.common.env_utils import make_vec_env
import torch as th

if __name__ == "__main__":
    env_id = "VNALL-v0"
    maml_model_path = "model/maml_sp500_20230102_131043.pth"
    algo_config = "configs/ppo_adapt.yaml"

    args = open_config(algo_config, env_id=env_id)
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
