import gymnasium as gym
import numpy as np

class DiscretizeAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env, num_bins: int = 5):
        super().__init__(env)
        self.discrete = np.linspace(env.action_space.low, env.action_space.high, num_bins)

    def action(self, action):
        return self.discrete[np.argmin(np.abs(self.discrete - action))]
    