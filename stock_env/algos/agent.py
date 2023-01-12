import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear
from typing import List
from copy import deepcopy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.xavier_normal_(layer.weight)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        hiddens: List[int] = [64, 64],
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        shared_layers = []
        last_layer_dim = np.array(envs.single_observation_space.shape).prod()
        for dim in hiddens:
            shared_layers.append(layer_init(nn.Linear(last_layer_dim, dim)))
            shared_layers.append(activation_fn())
            last_layer_dim = dim

        self.critic = nn.Sequential(
            *deepcopy(shared_layers), layer_init(nn.Linear(last_layer_dim, 1), std=1.0)
        )
        self.actor_mean = nn.Sequential(
            *deepcopy(shared_layers),
            layer_init(
                nn.Linear(last_layer_dim, np.prod(envs.single_action_space.shape)),
                std=0.01,
            )
        )
        self.actor_logstd = nn.Parameter(
            th.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return (
            action,
            self.critic(x),
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
        )

    def forward(self, x, action=None):
        return self.get_action_and_value(x, action)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        distribution = Normal(action_mean, action_std)
        if deterministic:
            return distribution.mean
        return distribution.sample()


class MetaAgent(MetaModule):
    def __init__(
        self,
        envs,
        hiddens: List[int] = [64, 64],
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        shared_layers = []
        last_layer_dim = np.array(envs.single_observation_space.shape).prod()
        for dim in hiddens:
            shared_layers.append(layer_init(MetaLinear(last_layer_dim, dim)))
            shared_layers.append(activation_fn())
            last_layer_dim = dim

        self.critic = MetaSequential(
            *deepcopy(shared_layers), layer_init(MetaLinear(last_layer_dim, 1), std=1.0)
        )
        self.actor_mean = MetaSequential(
            *shared_layers,
            layer_init(
                MetaLinear(last_layer_dim, np.prod(envs.single_action_space.shape)),
                std=0.01,
            )
        )
        self.actor_logstd = nn.Parameter(
            th.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x, params=None):
        return self.critic(x, params=self.get_subdict(params, "critic"))

    def get_action_and_value(self, x, action=None, params=None):
        action_mean = self.actor_mean(x, params=self.get_subdict(params, "actor_mean"))

        if params is None:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = params["actor_logstd"].expand_as(action_mean)

        action_std = th.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        value = self.critic(x, params=self.get_subdict(params, "critic"))
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, value, logprob.sum(1), entropy.sum(1)

    def get_action(self, x, params=None, deterministic=False):
        action_mean = self.actor_mean(x, params=self.get_subdict(params, "actor_mean"))

        if params is None:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = params["actor_logstd"].expand_as(action_mean)

        action_std = th.exp(action_logstd)
        distribution = Normal(action_mean, action_std)

        if deterministic:
            return distribution.mean
        return distribution.sample()
