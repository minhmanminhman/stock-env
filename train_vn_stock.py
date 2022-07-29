from stable_baselines3.dqn import DQN
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
    init_cash=100e3)

model = DQN(
    'MlpPolicy',
    env=env, 
    learning_rate=1e-3,
    gamma=0.999,
    buffer_size=50000,
    batch_size=128,
    train_freq=(4, "step"),
    gradient_steps=1,
    exploration_initial_eps=0.1,
    exploration_final_eps=0.1,
    learning_starts=0,
    target_update_interval=1000,
    tensorboard_log='log',
    verbose=1,
)
model.learn(
    total_timesteps=100000,
    eval_env=None,
    eval_freq=0,
    n_eval_episodes=0,
)

model.save('log/dqn_VietnamStock')

mean, std = evaluate_policy(model, env, n_eval_episodes=1)
print(f"Mean reward: {mean:.2f} +/- {std: .2f}")