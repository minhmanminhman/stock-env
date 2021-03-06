from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import stock_env
from stock_env.envs.vn_stock_env import VietnamStockEnv
import gym
import mt4_hst

df = mt4_hst.read_hst("stock_env/datasets/FPT1440.hst")
df = df[df['time'] >= '2012-01-01']
env = VietnamStockEnv(
    df=df,
    max_trade_lot=5,
    max_lot=10,
    init_cash=5e3)

# model = DQN(
#     'MlpPolicy',
#     env=env, 
#     learning_rate=1e-3,
#     gamma=0.999,
#     buffer_size=100000,
#     batch_size=128,
#     train_freq=(4, "step"),
#     gradient_steps=1,
#     exploration_initial_eps=1,
#     exploration_final_eps=0.1,
#     learning_starts=5000,
#     target_update_interval=1000,
#     tensorboard_log='log',
#     verbose=1,
# )
model = PPO(
    'MlpPolicy',
    env=env, 
    learning_rate=1e-3,
    n_steps=64,
    gamma=0.999,
    batch_size=32,
    tensorboard_log='log',
    verbose=0,
)
model.learn(
    total_timesteps=1000000,
    eval_env=None,
    eval_freq=0,
    n_eval_episodes=0,
)

model.save('log/ppo_VietnamStock')

mean, std = evaluate_policy(model, env, n_eval_episodes=1)
print(f"Mean reward: {mean:.2f} +/- {std: .2f}")

# run model to get detailed information in the enviroment
done = False
obs = env.reset()
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)

# get data
env.get_history().to_csv('temp/history.csv', index=False)