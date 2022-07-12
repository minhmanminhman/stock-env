from stock_env.envs.single_stock_env import SingleStockEnv
from stable_baselines3.dqn import DQN
import numpy as np
import pandas as pd

price = np.load('price.pkl.npy')
df = pd.DataFrame()
df['A'] = price
env = SingleStockEnv(df, init_cash=5e4)

model = DQN(
    'MlpPolicy',
    env=env, 
    learning_rate=1e-3,
    gamma=0.999,
    buffer_size=100000,
    batch_size=64,
    train_freq=(4, "step"),
    gradient_steps=1,
    # optimize_memory_usage=True,
    exploration_initial_eps=0.1,
    exploration_final_eps=0.1,
    learning_starts=100000,
    target_update_interval=1000,
    tensorboard_log='log',
    verbose=1,
)
model.learn(
    total_timesteps=1e6,
    eval_env=None,
    eval_freq=0,
    n_eval_episodes=0,
)