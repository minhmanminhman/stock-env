from stock_env.algos.mamlpp import MAMLPlusPlus
from stock_env.envs import *
from stock_env.common.common_utils import open_config
from stock_env.common.env_utils import make_vec_env

if __name__ == "__main__":
    env_id = "SP500-v0"
    test_env_id = "VNALL-v0"
    configs_path = "configs/mamlpp.yaml"

    args = open_config(configs_path, env_id=env_id)
    envs = make_vec_env(env_id, num_envs=args.num_envs)
    algo = MAMLPlusPlus(args, envs)
    try:
        algo.learn()
    except KeyboardInterrupt:
        pass
    finally:
        test_envs = make_vec_env(test_env_id, num_envs=args.num_envs)
        test_args = open_config(configs_path, env_id=test_env_id)
        algo.test(test_envs=test_envs, test_args=test_args)
        algo.close()
