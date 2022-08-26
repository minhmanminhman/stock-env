from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import stock_env
from stock_env.envs.multi_stock import MultiStockEnv, MultiStockContinuousEnv
from stable_baselines3.common.env_checker import check_env
from stock_env.feature.feature_extractor import *
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    env = 'MultiStockContinuousEnv'
    algo = 'ppo'
    tickers = "SSI VNM HPG PVD".split()
    path = "../stock_datasets/"
    feature_extractor = TrendFeatures()
    name = f"{algo}_{env}_{feature_extractor.__class__.__name__}"
    
    env = MultiStockContinuousEnv(
        tickers=tickers,
        feature_extractor=feature_extractor,
        data_folder_path=path,
        init_cash=500e3
    )
    check_env(env)
    
    model = PPO(
        'MlpPolicy',
        env=env, 
        learning_rate=1e-4,
        gamma=0.99,
        n_steps=16,
        ent_coef=0.01,
        normalize_advantage=True,
        tensorboard_log='log',
        verbose=0,
        policy_kwargs=dict(net_arch=[256, dict(vf=[512, 512])])
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
    obs = env.reset(current_tick=0)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
    histories, ticker_history = env.get_history()
    
    # get data
    histories.to_csv(f'temp/history/histories_{name}.csv', index=False)
    
    for ticker in tickers:
        ticker_history[ticker].to_csv(f'temp/history/ticker_history_{ticker}_{name}.csv', index=False)