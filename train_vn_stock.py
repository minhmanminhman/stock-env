from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stock_env.envs.vn_stock_env import *
from stock_env.feature.feature_extractor import *
import mt4_hst
from stable_baselines3.common.env_checker import check_env
from stock_env.wrappers.stack_obs import BufferWrapper

if __name__ == '__main__':
    env = 'VietnamStockContinuousEnv'
    algo = 'ppo'
    ticker = "VNM"
    path = "../stock_datasets/"
    feature_extractor = TrendFeatures()
    n_steps = 10
    name = f"{algo}_{env}_{feature_extractor.__class__.__name__}_{ticker}"
    
    env = VietnamStockContinuousEnv(
        df = mt4_hst.read_hst(path + ticker + '1440.hst'),
        feature_extractor = feature_extractor,
        init_cash = 100e3,
        ticker = ticker,
    )
    # env = BufferWrapper(env, n_steps)
    check_env(env)
    model = PPO(
        'MlpPolicy',
        env=env, 
        learning_rate=1e-3,
        ent_coef=0.01,
        n_steps=20,
        gamma=0.99,
        batch_size=20,
        tensorboard_log='log',
        verbose=0,
    )

    print(model.policy)
    model.learn(
        total_timesteps=100000,
        eval_env=None,
        eval_freq=0,
        n_eval_episodes=0,
    )

    model.save(f'log/{name}')

    mean, std = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"Mean reward: {mean:.2f} +/- {std: .2f}")

    # run model to get detailed information in the enviroment
    done = False
    obs = env.reset(current_tick=0)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

    # get data
    env.get_history().to_csv(f'temp/history/{name}.csv', index=False)