from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from stock_env.envs.multi_stock import MultiStockContinuousEnv
from stable_baselines3.common.env_checker import check_env
from stock_env.feature.feature_extractor import *
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    env = 'MultiStockContinuousEnv'
    algo = 'recurrentppo'
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
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env=env,
        learning_rate=1e-4,
        n_steps=50,
        batch_size=50,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log='log',
        policy_kwargs=dict(
            net_arch=[256, dict(vf=[512, 512])],
            n_lstm_layers=2,
            shared_lstm=True,
            enable_critic_lstm=False,
        ),
        verbose=1
    )

    
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
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, _, done, _ = env.step(action)
        episode_starts = done
    
    histories, ticker_history = env.get_history()
    
    # get data
    histories.to_csv(f'temp/history/histories_{name}.csv', index=False)
    
    for ticker in tickers:
        ticker_history[ticker].to_csv(f'temp/history/ticker_history_{ticker}_{name}.csv', index=False)