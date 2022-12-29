import gymnasium as gym
import torch as th
from collections import OrderedDict
import numpy as np
from typing import Dict, Union, List, Tuple, Any
import logging
from copy import deepcopy

from stock_env.envs import *
from stock_env.common.evaluation import evaluate_agent, play_an_episode
from stock_env.algos.agent import MetaAgent
from stock_env.common.common_utils import (
    open_config,
    create_performance,
)
from stock_env.algos.maml import adapt
from stock_env.algos.buffer import RolloutBuffer

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class ExperimentManager:
    def __init__(
        self,
        args_path: str,
        env_id: str,
        methods_state_dict: Dict[str, Any],
        device: str = "auto",
    ) -> None:

        self.args = open_config(args_path, env_id=env_id)
        self.env_id = env_id
        if device == "auto":
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        else:
            self.device = th.device(device)

        self.envs = self._make_envs(env_id=self.env_id, num_envs=self.args.num_envs)
        self.buffer = RolloutBuffer(
            num_steps=self.args.num_steps,
            envs=self.envs,
            device=self.device,
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
        )

        # create agents
        self.init_agent = MetaAgent(envs=self.envs, hiddens=self.args.hiddens).to(
            self.device
        )
        self.methods_state_dict = methods_state_dict
        self.meta_params = None
        self._parse_meta_params()

    def _make_envs(self, env_id, num_envs):
        return MetaVectorStockEnv([lambda: gym.make(env_id) for _ in range(num_envs)])

    def _parse_meta_params(self):
        self.meta_params = {}
        for method, path in self.methods_state_dict.items():
            if method == "random":
                self.meta_params[method] = self.init_agent.state_dict()
            else:
                self.meta_params[method] = OrderedDict(th.load(path))

    def _evaluate_agent(
        self, agent: MetaAgent, n_eval_episodes: int = None
    ) -> Tuple[float, float]:
        if n_eval_episodes is None:
            n_eval_episodes = self.args.n_eval_episodes
        mean, std = evaluate_agent(
            agent=agent, envs=self.envs, n_eval_episodes=n_eval_episodes
        )
        return mean, std

    def _adapt_one_step(
        self,
        agent: MetaAgent,
        step_size: float = 0.1,
    ) -> None:
        params, inner_loss = adapt(
            args=self.args,
            meta_agent=agent,
            meta_env=self.envs,
            buffer=self.buffer,
            step_size=step_size,
            n_adapt_steps=1,
        )
        return params, inner_loss

    def mass_adaption_results(
        self,
        methods: List[str],
        maybe_num_tasks: Union[int, List[str]] = 5,
        total_adapt_steps: int = 5,
        n_eval_episodes: int = None,
    ):
        adaption_results = {
            "task": [],
            "n_adapt_steps": [],
            "model_type": [],
            "mean": [],
            "std": [],
            "inner_loss": [],
        }
        if type(maybe_num_tasks) == int:
            num_tasks = maybe_num_tasks
            tasks = self.envs.sample_task(num_tasks)
        elif type(maybe_num_tasks) == list:
            tasks = maybe_num_tasks

        for method in methods:
            for task in tasks:
                self.envs.reset_task(task)
                logging.info(f"Evaluating ticker {task} with method '{method}'...")
                assert method in self.meta_params.keys(), f"method '{method}' not found"
                agent = deepcopy(self.init_agent)
                agent.load_state_dict(self.meta_params[method])

                for n_adapt_steps in range(total_adapt_steps + 1):
                    mean, std = self._evaluate_agent(
                        agent=agent, n_eval_episodes=n_eval_episodes
                    )

                    if n_adapt_steps > 0:
                        step_size = self.args.inner_lr * 0.5
                    else:
                        step_size = self.args.inner_lr
                    params, inner_loss = self._adapt_one_step(agent, step_size)
                    agent.load_state_dict(params)

                    adaption_results["task"].append(task)
                    adaption_results["n_adapt_steps"].append(n_adapt_steps)
                    adaption_results["model_type"].append(method)
                    adaption_results["mean"].append(mean)
                    adaption_results["std"].append(std)
                    adaption_results["inner_loss"].append(inner_loss)
        return pd.DataFrame(adaption_results)

    def mass_trading_performance(
        self,
        methods: List[str],
        maybe_num_tasks: Union[int, List[str]] = 5,
        total_adapt_steps: int = 5,
    ) -> None:

        eval_env = self._make_envs(env_id=self.env_id, num_envs=1)
        eval_env.train(False)
        if type(maybe_num_tasks) == int:
            num_tasks = maybe_num_tasks
            tasks = self.envs.sample_task(num_tasks)
        elif type(maybe_num_tasks) == list:
            tasks = maybe_num_tasks

        metrics = [
            "annual_return",
            "cum_returns_final",
            "sharpe_ratio",
            "max_drawdown",
            "annual_volatility",
            "value_at_risk",
        ]
        output_params = {}
        array_adapt_steps = np.arange(0, total_adapt_steps + 1)

        perf_df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [array_adapt_steps, tasks, metrics],
                names=["n_adapt_steps", "ticker", "metric"],
            ),
            columns=methods,
        )
        for method in methods:
            assert method in self.meta_params.keys(), f"method '{method}' not found"

        for method in methods:
            output_params[method] = {}
            for task in tasks:
                self.envs.reset_task(task)
                eval_env.reset_task(task)
                agent = deepcopy(self.init_agent)
                agent.load_state_dict(self.meta_params[method])
                l_params = []
                for ith_adapt in range(total_adapt_steps + 1):

                    logging.info(
                        f"Adapt #{ith_adapt}: trading '{task}' with method '{method}'..."
                    )
                    if ith_adapt > 0:
                        step_size = self.args.inner_lr * 0.5
                    else:
                        step_size = self.args.inner_lr
                    params, _ = self._adapt_one_step(agent, step_size)
                    l_params.append(params)
                    agent.load_state_dict(params)

                    info = play_an_episode(
                        agent=agent, envs=eval_env, device=self.device
                    )
                    df = info["final_info"][0]["final_history"]
                    returns = df.set_index("time")["portfolio_value"].pct_change()
                    perf = create_performance(returns, plot=False)

                    for metric, value in perf.items():
                        perf_df.loc[(ith_adapt, task, metric), method] = round(value, 5)
                # save params
                output_params[method][task] = l_params
        return perf_df, output_params
