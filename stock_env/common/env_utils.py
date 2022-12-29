from typing import Tuple, Union, List, Dict, Any, Optional, Type, Callable
import numpy as np
import torch as th
from gymnasium import spaces
import gymnasium as gym


def get_obs_shape(observation_space: spaces.Space) -> Tuple[int, ...]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space: (spaces.Space)
    :return: (Tuple[int, ...])
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    else:
        raise NotImplementedError()


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space: (spaces.Space)
    :return: (int)
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError()


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.
    :param device: (Union[str, th.device]) One for 'auto', 'cuda', 'cpu'
    :return: (th.device)
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device == th.device("cuda") and not th.cuda.is_available():
        return th.device("cpu")

    return device


def unwrap_wrapper(
    env: gym.Env, wrapper_class: Type[gym.Wrapper]
) -> Optional[gym.Wrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: Type[gym.Env], wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def make_wrapped_env(name, gamma=0.99):
    def _thunk():
        env = gym.make(name)
        if not is_wrapped(env, gym.wrappers.RecordEpisodeStatistics):
            env = gym.wrappers.RecordEpisodeStatistics(env)
        if not is_wrapped(env, gym.wrappers.ClipAction):
            env = gym.wrappers.ClipAction(env)
        if not is_wrapped(env, gym.wrappers.NormalizeObservation):
            env = gym.wrappers.NormalizeObservation(env)
        if not is_wrapped(env, gym.wrappers.TransformObservation):
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10)
            )
        if not is_wrapped(env, gym.wrappers.NormalizeReward):
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if not is_wrapped(env, gym.wrappers.TransformReward):
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )
        return env

    return _thunk
