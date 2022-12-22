import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces


class StackAndSkipObs(gym.Wrapper):
    def __init__(self, env, num_stack, num_skip, dtype=np.float32):
        super(StackAndSkipObs, self).__init__(env)
        self.dtype = dtype
        self.num_stack = num_stack
        self.num_skip = num_skip
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(num_stack), old_space.high.repeat(num_stack)
        )
        self.obs_buffer = deque(maxlen=num_stack)

    def reset(self):
        obs, info = self.env.reset()
        [self.obs_buffer.append(obs) for _ in range(self.num_stack)]
        return self._get_observation(), info

    def step(self, action):
        total_reward, terminated, truncated, info = 0.0, False, False, {}
        for _ in range(self.num_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.obs_buffer.append(observation)

            if terminated or truncated:
                break
        return self._get_observation(), total_reward, terminated, truncated, info

    def _get_observation(self):
        return spaces.flatten(self.observation_space, self.obs_buffer)
