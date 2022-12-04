from gymnasium.vector import SyncVectorEnv
import torch as th
import numpy as np
from typing import NamedTuple

class RolloutBuffer:
    
    class RolloutBufferSamples(NamedTuple):
        obs: th.Tensor
        actions: th.Tensor
        values: th.Tensor
        logprobs: th.Tensor
        advantages: th.Tensor
        returns: th.Tensor
    
    def __init__(
        self,
        num_steps: int,
        envs: SyncVectorEnv,
        device: str = 'cpu',
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.device = device
        self.num_steps = num_steps
        self.num_envs = envs.num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.obs = th.zeros((num_steps, self.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = th.zeros((num_steps, self.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = th.zeros((num_steps, self.num_envs)).to(device)
        self.rewards = th.zeros((num_steps, self.num_envs)).to(device)
        self.dones = th.zeros((num_steps, self.num_envs)).to(device)
        self.values = th.zeros((num_steps, self.num_envs)).to(device)

    def add(self, index, obs, actions, logprobs, rewards, dones, values):
        self.obs[index] = obs
        self.actions[index] = actions
        self.logprobs[index] = logprobs
        self.rewards[index] = rewards
        self.dones[index] = dones
        self.values[index] = values

    def compute_returns(self, next_value, done):
        self.advantages = th.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

    def get(self, batch_size: int = None):
        indices = np.random.permutation(self.num_steps)
        
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.num_steps

        start_idx = 0
        while start_idx < self.num_steps:
            _indices = indices[start_idx:start_idx + batch_size]
            yield self.RolloutBufferSamples(
                obs=self.obs[_indices],
                actions=self.actions[_indices],
                values=self.values[_indices].flatten(),
                logprobs=self.logprobs[_indices].flatten(),
                advantages=self.advantages[_indices].flatten(),
                returns=self.returns[_indices].flatten(),
            )
            start_idx += batch_size