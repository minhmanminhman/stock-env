from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from stock_env.envs.vn_stock_env import VietnamStockContinuousEnv
from stable_baselines3.common.env_checker import check_env
from stock_env.feature.feature_extractor import *
import mt4_hst
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    env = 'VietnamStockContinuousEnv'
    algo = 'recurrentppo'
    ticker = "SSI"
    path = "../stock_datasets/"
    feature_extractor = TrendFeatures()
    name = f"{algo}_{env}_{feature_extractor.__class__.__name__}_{ticker}"
    
    env = VietnamStockContinuousEnv(
        df = mt4_hst.read_hst(path + ticker + '1440.hst'),
        feature_extractor = feature_extractor,
        init_cash = 100e3,
        ticker = ticker,
    )
    check_env(env)
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env=env,
        learning_rate=1e-4,
        n_steps=20,
        batch_size=20,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log='log',
        policy_kwargs=dict(
            net_arch=[64, dict(vf=[64, 64])],
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
    
    # get data
    env.get_history().to_csv(f'temp/history/{name}.csv', index=False)