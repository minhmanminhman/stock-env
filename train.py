from stable_baselines3.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import stock_env
import gym

env = gym.make('SimulatorStock-v0')

model = DQN(
    'MlpPolicy',
    env=env, 
    learning_rate=1e-3,
    gamma=0.999,
    buffer_size=100000,
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
trained_model = model.learn(
    total_timesteps=500000,
    eval_env=None,
    eval_freq=0,
    n_eval_episodes=0,
)

model.save('log/dqn_SimulatorStock-v0')

mean, std = evaluate_policy(trained_model, env, n_eval_episodes=10)
print(f"Mean reward: {mean:.2f} +/- {std: .2f}")