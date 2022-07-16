from stock_env.envs.single_stock_env import SingleStockEnv
from stable_baselines3.dqn import DQN
import numpy as np
import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy

price = np.load('price.pkl.npy')
df = pd.DataFrame()
df['A'] = price
env = SingleStockEnv(
    df=df, 
    init_cash=5e4,
)

model = DQN(
    'MlpPolicy',
    env=env, 
    learning_rate=1e-3,
    gamma=0.999,
    buffer_size=10000,
    batch_size=128,
    train_freq=(10, "step"),
    gradient_steps=1,
    # optimize_memory_usage=True,
    exploration_initial_eps=0.1,
    exploration_final_eps=0.1,
    learning_starts=0,
    target_update_interval=1000,
    tensorboard_log='log',
    verbose=1,
)
trained_model = model.learn(
    total_timesteps=100000,
    eval_env=None,
    eval_freq=0,
    n_eval_episodes=0,
)

model.save('log/dqn_single_stock')

mean, std = evaluate_policy(trained_model, env, n_eval_episodes=1)
print(f"Mean reward: {mean:.2f} +/- {std: .2f}")