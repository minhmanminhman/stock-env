import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear
from typing import List
from copy import deepcopy
from itertools import zip_longest

# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     th.nn.init.xavier_normal_(layer.weight)
#     th.nn.init.constant_(layer.bias, bias_const)
#     return layer


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # th.nn.init.orthogonal_(layer.weight, std)
    # th.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        hiddens: List[int] = [64, 64],
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.low, self.high = (
            envs.single_action_space.low.item(),
            envs.single_action_space.high.item(),
        )
        last_layer_dim = np.array(envs.single_observation_space.shape).prod()

        shared_net, policy_net, value_net = [], [], []
        for _, layer in enumerate(hiddens):
            # If shared layer
            if isinstance(layer, int):
                shared_net.append(layer_init(nn.Linear(last_layer_dim, layer)))
                shared_net.append(activation_fn())
                last_layer_dim = layer
            else:
                assert isinstance(
                    layer, dict
                ), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(
                        layer["pi"], list
                    ), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(
                        layer["vf"], list
                    ), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim
        last_layer_dim_vf = last_layer_dim

        # Build the non-shared part of the network
        for _, (pi_layer_size, vf_layer_size) in enumerate(
            zip_longest(policy_only_layers, value_only_layers)
        ):
            if pi_layer_size is not None:
                assert isinstance(
                    pi_layer_size, int
                ), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(
                    layer_init(nn.Linear(last_layer_dim_pi, pi_layer_size))
                )
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(
                    vf_layer_size, int
                ), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(
                    layer_init(nn.Linear(last_layer_dim_vf, vf_layer_size))
                )
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        self.shared_net = nn.Sequential(*shared_net)
        value_net.append(layer_init(nn.Linear(last_layer_dim_vf, 1), std=1.0))
        self.critic = nn.Sequential(*value_net)
        policy_net.append(
            layer_init(
                nn.Linear(last_layer_dim_pi, np.prod(envs.single_action_space.shape)),
                std=0.01,
            )
        )
        self.actor_mean = nn.Sequential(*policy_net)
        self.actor_logstd = nn.Parameter(
            th.zeros(1, np.prod(envs.single_action_space.shape)),
            requires_grad=True,
        )

    def get_value(self, x):
        return self.critic(self.shared_net(x))

    def get_action_and_value(self, x, action=None):
        x = self.shared_net(x)
        action_mean = self.actor_mean(x)
        action_std = th.ones_like(action_mean) * self.actor_logstd.exp()
        try:
            probs = Normal(action_mean, action_std)
            # print(probs)
        except:
            print("action_std", action_std)
            raise
        if action is None:
            action = probs.sample()
        action = self._adjust_actions(action)
        return (
            action,
            self.critic(x),
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
        )

    def forward(self, x, action=None):
        return self.get_action_and_value(x, action)

    def get_action(self, x, deterministic=False):
        x = self.shared_net(x)
        action_mean = self.actor_mean(x)
        action_std = th.ones_like(action_mean) * self.actor_logstd.exp()
        distribution = Normal(action_mean, action_std)
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()
        action = self._adjust_actions(action)
        return action

    def _adjust_actions(self, actions):
        fixed_values = (
            th.linspace(self.low, self.high, steps=5)
            .view(1, -1)
            .repeat(actions.shape[0], 1)
        )
        chose_idx = th.argmin(th.abs(actions - fixed_values), dim=1).view(-1, 1)
        adjusted_actions = fixed_values.gather(1, chose_idx)
        return adjusted_actions


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
