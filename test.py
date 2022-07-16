from statistics import mode
from stock_env.envs.single_stock_env import SingleStockEnv
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy

price = np.load('price.pkl.npy')
df = pd.DataFrame()
df['A'] = price
env = SingleStockEnv(
    df=df,
    init_cash=5e4,
)
check_env(env)
# print(env.observation_space)
# print(env.observation_space.sample())

model = DQN.load("log/dqn_single_stock", env=env)
mean, std = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
# print(f"Mean reward: {mean:.2f} +/- {std: .2f}")

done = False
obs = env.reset()
while not done:
    action, _ = model.predict(obs, deterministic=True)
    # print(action)
    obs, _, done, _ = env.step(action)

data = pd.concat([df, pd.DataFrame(env.history)], axis=1, join='inner')
data.to_csv('history.csv')
print(data.tail(20))