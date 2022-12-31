from abc import abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from typing import Tuple, Type
from dataclasses import dataclass

from ..data_loader import BaseDataLoader


@dataclass
class Position:
    t0_quantity: int = 0
    t1_quantity: int = 0
    t2_quantity: int = 0
    on_hand: int = 0

    @property
    def quantity(self):
        quantity = self.t0_quantity + self.t1_quantity + self.t2_quantity + self.on_hand
        return quantity

    def update_position(self):
        self.on_hand += self.t2_quantity
        self.t2_quantity = self.t1_quantity
        self.t1_quantity = self.t0_quantity
        self.t0_quantity = 0

    def transact_trade(self, delta_shares):
        if delta_shares >= 0:  # long or hold
            self.t0_quantity = delta_shares
        else:  # short
            self.on_hand = self.on_hand + delta_shares
            assert self.on_hand >= 0

    def reset(self):
        self.t0_quantity = self.t1_quantity = self.t2_quantity = self.on_hand = 0


class BaseVietnamStockEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data_loader: Type[BaseDataLoader],
        init_cash: float = 2e4,
        random_seed: int = None,
    ):
        self.seed(random_seed)
        self.data_loader = data_loader
        self.cash = self.init_cash = init_cash
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _update_history(self, info: dict) -> None:
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError
