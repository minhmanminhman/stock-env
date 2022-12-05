import pandas as pd
import torch as th
import gymnasium as gym
from stock_env.envs import *
from stock_env.common.evaluation import play_an_episode
from stock_env.algos.agent import MetaAgent

if __name__ == '__main__':
    
    task_env = MetaVectorEnv([lambda: gym.make('FAANGTask-v0') for _ in range(1)])
    agent = MetaAgent(task_env)
    state_dict = th.load('/Users/manbnm/stock-env/model/maml_2022-12-04_15-00-17.pth')
    agent.load_state_dict(state_dict)
    agent.eval()
    task_env.train(False)
    
    list_task = task_env.sample_task(1)
    print(f"Ticker: {list_task[0]}")
    task_env.reset_task(list_task[0])

    # run model to get detailed information in the enviroment
    envs = task_env
    obs, _ = envs.reset()
    obs = th.Tensor(obs)
    dones = np.zeros(envs.num_envs, dtype="bool")
    i = 0

    while not any(dones):
        actions = agent.get_action(obs)
        obs, reward, terminated, truncated, info = envs.step(actions.cpu().numpy())
        dones = terminated | truncated
        obs = th.Tensor(obs)
        if any(dones):
            print(info)
            