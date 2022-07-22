import pandas as pd
from stable_baselines3.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import stock_env

env = gym.make('SingleStock-v0')

model = DQN.load("log/dqn_single_stock", env=env)
mean, std = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
print(f"Mean reward: {mean:.2f} +/- {std: .2f}")

done = False
obs = env.reset()
while not done:
    action, _ = model.predict(obs, deterministic=True)
    # print(action)
    obs, _, done, _ = env.step(action)
# print(env.history)
data = pd.concat([env.df, pd.DataFrame(env.history, index=env.df.index)], axis=1, join='inner')
data.to_csv('history.csv')
print(data.tail(20))