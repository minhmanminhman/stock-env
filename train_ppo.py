from stock_env.algos.ppo import PPO
from stock_env.algos.agent import Agent
from stock_env.envs import *
from stock_env.common.common_utils import open_config
from stock_env.common.env_utils import make_wrapped_env


if __name__ == "__main__":
    try:
        env_id = "HalfCheetahDir-v0"
        args = open_config("configs/ppo.yaml", env_id=env_id)
        envs = MetaVectorEnv(
            [make_wrapped_env(env_id, args.gamma) for _ in range(args.num_envs)]
        )
        agent = Agent(envs, hiddens=args.hiddens)
        algo = PPO(args, envs, agent)
        algo.learn()
    except KeyboardInterrupt:
        pass
    finally:
        algo.test()
        algo.close()
