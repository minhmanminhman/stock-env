from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import stock_env
from stock_env.envs.multi_stock import MultiStockEnv, MultiStockContinuousEnv
from stable_baselines3.common.env_checker import check_env
from stock_env.feature.feature_extractor import *

if __name__ == '__main__':
    env = 'MultiStockContinuousEnv'
    algo = 'ppo'
    tickers = "FPT SSI VNM".split()
    path = "../stock_datasets/"
    feature_extractor = TrendFeatures()
    name = f"{algo}_{env}_{feature_extractor.__class__.__name__}"
    
    env = MultiStockContinuousEnv(
        tickers=tickers,
        feature_extractor=feature_extractor,
        data_folder_path=path)
    check_env(env)
    
    model = PPO(
        'MlpPolicy',
        env=env, 
        learning_rate=1e-3,
        gamma=0.999,
        n_steps=32,
        ent_coef=0.01,
        normalize_advantage=True,
        tensorboard_log='log',
        verbose=1,
        policy_kwargs=dict(net_arch=[128, dict(vf=[256, 256])])
    )
    
    model.learn(
        total_timesteps=1000000,
        eval_env=None,
        eval_freq=0,
        n_eval_episodes=0,
    )

    model.save(f'log/{name}')

    mean, std = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"Mean reward: {mean:.2f} +/- {std: .2f}")

    # run model to get detailed information in the enviroment
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
    histories, ticker_history = env.get_history()
    
    # get data
    histories.to_csv(f'temp/history/histories_{name}.csv', index=False)
    
    for ticker in tickers:
        ticker_history[ticker].to_csv(f'temp/history/ticker_history_{ticker}_{name}.csv', index=False)