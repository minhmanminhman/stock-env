import gymnasium as gym
import numpy as np

class StackObs(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(StackObs, self).__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps
        old_space = env.observation_space
        # self.observation_space = gym.spaces.Box(
        #     old_space.low.repeat(n_steps).reshape(n_steps, -1),
        #     old_space.high.repeat(n_steps).reshape(n_steps, -1))
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps), old_space.high.repeat(n_steps))
        self.buffer = np.zeros_like(
            self.observation_space.low.reshape(n_steps, -1), dtype=self.dtype)

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer.flatten()