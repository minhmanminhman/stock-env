from stock_env.algos.maml import MAML
from stock_env.algos.agent import MetaAgent, Agent
from stock_env.envs import *
from stock_env.common.common_utils import open_config
from stock_env.common.env_utils import make_vec_env


if __name__ == "__main__":
    env_id = "SP500-v0"
    test_env_id = "VNALL-v0"
    # env_id = "MiniFAANG-v0"
    # test_env_id = "MiniVNStock-v0"

    args = open_config("configs/maml.yaml", env_id=env_id)
    envs = make_vec_env(env_id, num_envs=args.num_envs)
    agent = Agent(envs, hiddens=args.hiddens)
    algo = MAML(args, envs, agent)

    try:
        algo.learn()
    except KeyboardInterrupt:
        pass
    finally:
        test_envs = make_vec_env(test_env_id, num_envs=args.num_envs)
        test_args = open_config("configs/maml.yaml", env_id=test_env_id)
        algo.test(test_envs=test_envs, test_args=test_args)
        algo.close()
