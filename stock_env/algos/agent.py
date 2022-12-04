import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torchmeta.modules import (
    MetaModule, 
    MetaSequential,
    MetaLinear
)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(th.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        
        return action, self.critic(x), probs.log_prob(action).view(-1), probs.entropy().view(-1)
    
    def get_action(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        return Normal(action_mean, action_std).sample()

class MetaAgent(MetaModule):
    def __init__(self, envs):
        super().__init__()
        self.critic = MetaSequential(
            layer_init(MetaLinear(np.array(envs.single_observation_space.shape).prod(), 4)),
            nn.Tanh(),
            layer_init(MetaLinear(4, 1), std=1.0),
        )
        self.actor_mean = MetaSequential(
            layer_init(MetaLinear(np.array(envs.single_observation_space.shape).prod(), 4)),
            nn.Tanh(),
            layer_init(MetaLinear(4, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(th.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x, params=None):
        return self.critic(x, params=self.get_subdict(params, 'critic'))

    def get_action_and_value(self, x, action=None, params=None):
        action_mean = self.actor_mean(x, params=self.get_subdict(params, 'actor_mean'))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        value = self.critic(x, params=self.get_subdict(params, 'critic'))
        return action, value, probs.log_prob(action).view(-1), probs.entropy().view(-1)
    
    def get_action(self, x, params=None):
        action_mean = self.actor_mean(x, params=self.get_subdict(params, 'actor_mean'))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        return Normal(action_mean, action_std).sample()