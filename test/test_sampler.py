import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch as th
import numpy as np
from stock_env.algos.agent import Agent

def make_env(env_id, seed, gamma):
    def thunk():
        env = gym.make(env_id, render_mode='human')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == '__main__':    

    envs = SyncVectorEnv([make_env('MountainCarContinuous-v0', None, 0.999) for _ in range(1)])
    agent = Agent(envs)
    agent.load_state_dict(th.load("/Users/manbnm/stock-env/model/mountain_car_agent.pth"))
    # env = gym.make('MountainCarContinuous-v0', render_mode='human')

    obs, infos = envs.reset()

    # play the environment
    while True:
        action = agent.get_action(th.Tensor(obs))
        obs, rewards, term, trunc, infos = envs.step(action)
        envs.envs[0].unwrapped.render()
        if any(term | trunc):
            break