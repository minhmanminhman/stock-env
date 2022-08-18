import numpy as np
from gym import spaces
import gym
import pandas as pd
import mt4_hst
from .vn_stock_env import VietnamStockEnv
from ..feature.feature_extractor import BaseFeaturesExtractor

class MultiStockEnv(gym.Env):
    def __init__(
        self, 
        tickers: str, 
        feature_extractor: BaseFeaturesExtractor,
        data_folder_path: str,
        tick_size: float = 0.05,
        lot_size: int = 100,
        max_trade_lot: int = 5,
        max_lot: int = 10,
        kappa: float = 1e-4,
        init_cash: float = 2e4,
        random_seed: int = None,
    ):
        self.tickers = tickers
        self.feature_extractor = feature_extractor
        self.data_folder_path = data_folder_path
        self.n_ticker = len(self.tickers)
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_trade_lot = max_trade_lot
        self.max_lot = max_lot
        self.kappa = kappa
        self.init_total_cash = init_cash
        self.init_ticker_cash = self.init_total_cash / self.n_ticker
        self.random_seed = random_seed
        
        self._create_envs()
        self.obs_dim = sum([env.observation_space.shape[0] for env in self.envs])
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim + 1,),
            dtype=np.float64
        )
        self.action_space = spaces.MultiDiscrete([env.action_space.n for env in self.envs])
        
        self._start_tick = 0
        self._end_tick = self.df.shape[0] - 1
        self._current_tick = None
        self.done = None
        self.total_reward = None
        self.positions = None
        self.histories = None
    
    def _create_envs(self):
        self.envs = []
        df = pd.DataFrame()
        for ticker in self.tickers:
            _df = mt4_hst.read_hst(self.data_folder_path + ticker + "1440.hst")
            _df.sort_values(by='time', inplace=True)
            _df['ticker'] = ticker
            df = pd.concat([df, _df])

        # align date
        df = df.set_index('time', drop=False).pivot(columns=['ticker'])
        # format axis
        self.df = df.swaplevel(0, 1, axis=1).sort_index(axis=1).reset_index(drop=True)
        self.df.dropna(inplace=True)
        
        for ticker in self.tickers:
            _env = VietnamStockEnv(
                df = self.df[ticker],
                feature_extractor = self.feature_extractor,
                tick_size = self.tick_size,
                lot_size = self.lot_size,
                max_trade_lot = self.max_trade_lot,
                max_lot = self.max_lot,
                kappa = self.kappa,
                init_cash = self.init_ticker_cash,
                random_seed = self.random_seed,
                ticker = ticker,
            )
            self.envs.append(_env)
        self.df, _ = self.df.align(self.envs[0].df, join='inner', axis=0)
    
    @property
    def nav(self):
        return sum([env.nav for env in self.envs])
    
    @property
    def cash(self):
        return sum([env.cash for env in self.envs])
    
    @property
    def portfolio_value(self):
        return self.nav + self.cash
    
    def reset(self, current_tick=0):
        if current_tick == 'random':
            self._current_tick = np.random.randint(self._start_tick, self._end_tick)
        elif isinstance(current_tick, int):
            self._current_tick = current_tick
        else:
            raise NotImplementedError
        self.histories = {
            'total_portfolio_value': [],
            'total_cash': [],
            'total_nav': [],
            'total_step_reward': [],
            'time': [],
        }
        observations = [env.reset(current_tick=self._current_tick) for env in self.envs]
        return self._get_observation(observations)
    
    def _get_observation(self, observations: list):
        obs = np.concatenate(observations)
        obs = np.append(obs, self.cash).astype(np.float64)
        return obs
    
    def step(self, action: np.ndarray):
        obs, step_reward, done = [], 0, False
        for _action, env in zip(action, self.envs):
            _obs, _reward, _done, _ = env.step(_action)
            obs.append(_obs)
            step_reward += _reward
            done = done or _done
        obs = self._get_observation(obs)
        
        # always update history last
        info = dict(
            total_portfolio_value = self.portfolio_value,
            total_nav = self.nav,
            total_cash = self.cash,
            total_step_reward = step_reward,
            time = env.df.time.iloc[env._current_tick],
        )
        self._update_history(info)
        return obs, step_reward, done, info
    
    def get_history(self):
        history_df = pd.DataFrame(self.histories)
        ticker_history = {}
        for env in self.envs:
            ticker_history[env.ticker] = env.get_history()
        return history_df, ticker_history
    
    def _update_history(self, info: dict) -> None:
        if not self.histories:
            self.histories = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.histories[key].append(value)