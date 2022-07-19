"""
Ornstein-Uhlenbeck Process Simulator
    dXt = alpha * (gamma - Xt)dt + beta * dWt
"""
from dataclasses import dataclass
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class OUParams:
    alpha: float = np.log(2) / 5 # mean reversion parameter
    beta: float = 0.1 # Brownian motion scale (standard deviation)
    gamma: float = 0 # asymptotic mean

class OUSimulator:
    """
    Simulate price according to paper Machine Learning for Trading
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3015609
    """
    def __init__(
        self,
        size: int,
        equilibrium_price: float,
        seed: int,
        params: OUParams,
    ) -> None:
        self.size = size
        self.pe = equilibrium_price
        self.seed = seed
        self.alpha = params.alpha
        self.beta = params.beta
        self.gamma = params.gamma
        self.dt = 1 / self.size
    
    def _transform_price(self, process):
        return np.exp(process) * self.pe
    
    def _eucler_simulator(self):
        npr.seed(self.seed)
        process = np.zeros(self.size, dtype=np.float64)
        for i in range(1, self.size):
            process[i] = process[i-1] \
                        + self.alpha * (self.beta - process[i-1]) * self.dt \
                        + self.beta * np.sqrt(self.dt) * npr.normal()
        return process
    
    def simulate(self):
        self.process = self._eucler_simulator()
        self.price = self._transform_price(self.process)
        return self.price