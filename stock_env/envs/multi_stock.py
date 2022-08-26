import numpy as np
from gym import spaces
import gym
import pandas as pd
import mt4_hst
from .vn_stock_env import VietnamStockContinuousEnv, VietnamStockEnv
from ..feature.feature_extractor import BaseFeaturesExtractor

class MultiStockEnv(gym.Env):
    def __init__(
        self, 
        tickers: str, 
        feature_extractor: BaseFeaturesExtractor,
        data_folder_path: str,
        init_cash: float = 2e4,
        env_kwargs: dict = {},
    ):
        self.tickers = tickers
        self.feature_extractor = feature_extractor
        self.data_folder_path = data_folder_path
        self.n_ticker = len(self.tickers)
        self.env_kwargs = env_kwargs
        self.init_total_cash = init_cash
        self.init_ticker_cash = self.init_total_cash / self.n_ticker
        
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
        self.df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
        self.df.dropna(inplace=True)
        self.df = self.df.reset_index(drop=True)
        
        for ticker in self.tickers:
            _env = VietnamStockEnv(
                df = self.df[ticker],
                feature_extractor = self.feature_extractor,
                init_cash = self.init_ticker_cash,
                ticker = ticker,
                ** self.env_kwargs
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
    
    def reset(self, current_tick="random"):
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


class MultiStockContinuousEnv(gym.Env):
    def __init__(
        self, 
        tickers: str, 
        feature_extractor: BaseFeaturesExtractor,
        data_folder_path: str,
        init_cash: float = 2e4,
        env_kwargs: dict = {},
    ):
        self.tickers = tickers
        self.feature_extractor = feature_extractor
        self.data_folder_path = data_folder_path
        self.n_ticker = len(self.tickers)
        self.env_kwargs = env_kwargs
        self.init_total_cash = init_cash
        self.init_ticker_cash = self.init_total_cash / self.n_ticker
        
        self._create_envs()
        self.obs_dim = sum([env.observation_space.shape[0] for env in self.envs])
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim + 1,),
            dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(len(self.envs),),
            dtype=np.float32
        )
        
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
        self.df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
        self.df.dropna(inplace=True)
        self.df = self.df.reset_index(drop=True)

        for ticker in self.tickers:
            _env = VietnamStockContinuousEnv(
                df = self.df[ticker],
                feature_extractor = self.feature_extractor,
                init_cash = self.init_ticker_cash,
                ticker = ticker,
                ** self.env_kwargs
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
    
    def _get_observation(self, observations: list):
        obs = np.concatenate(observations)
        cash_percent = self.cash / self.portfolio_value
        obs = np.append(obs, cash_percent).astype(np.float32)
        return obs
    
    def reset(self, current_tick="random"):
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
        self.n_steps = 0
        observations = [env.reset(current_tick=self._current_tick) for env in self.envs]
        return self._get_observation(observations)
    
    def step(self, action: np.ndarray):
        obs, step_reward, done = [], 0, False
        self.n_steps += 1
        action = self.unscale_action(action)
        action = action.reshape(-1, 1)
        for _action, env in zip(action, self.envs):
            _obs, _reward, _done, _ = env.step(_action)
            obs.append(_obs)
            step_reward += _reward
            done = done or _done
        obs = self._get_observation(obs)
        
        step_reward = self._reward()
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
    
    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = 0, 1
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

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
    
    def _reward(self):
        eps = 1e-8
        cum_log_return = np.log((self.portfolio_value + eps) / (self.init_total_cash + eps))
        reward = cum_log_return / self.n_steps
        return reward