import gym
from stock_env.envs.single_stock_env import SingleStockEnv
from stock_env.simulate_price import OUParams, OUSimulator

class SimulatorStockEnv(gym.Env):
    def __init__(
        self, 
        size: int,
        equilibrium_price: float = 50,
        params: OUParams = OUParams,
        seed: int = None,
        env_params: dict = {}
    ):
        self.size = size
        self.equilibrium_price = equilibrium_price
        self.seed = seed
        self.params = params
        self.env_params = env_params
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        df = OUSimulator(
            size=self.size,
            equilibrium_price=self.equilibrium_price,
            seed=self.seed,
            params=self.params
        )
        self.env = SingleStockEnv(df=df, **self.env_params)
        return self.env.reset()