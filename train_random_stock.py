from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stock_env.envs import *
from stock_env.feature import feature_extractor
from stock_env.feature.feature_extractor import *
from stock_env.wrappers import StackObs
from stock_env.envs import RandomStockEnv
from stock_env.data_loader import RandomStockLoader

def name_generate(env, algo, feature_extractor, ticker, postfix=None):
    env_name = env.__class__.__name__
    algo_name = algo.__class__.__name__
    extractor_name = feature_extractor.__class__.__name__
    name = f"{algo_name}_{env_name}_{extractor_name}"
    if type(ticker) == str:
        name += f"_{ticker}"
    if postfix is not None:
        name += f"_{postfix}"
    return name

if __name__ == '__main__':
    path = "../stock_datasets/"
    tickers = "SSI VND HCM MBS VCI".split()
    postfix = "finservice"
    n_steps = 5
    feature_extractor = TrendFeatures
    
    data_loader = RandomStockLoader(
        tickers = tickers,
        data_folder_path = "../stock_datasets/",
        feature_extractor = TrendFeatures
    )

    env = RandomStockEnv(data_loader)
    env = StackObs(env, 5)
    
    model = PPO(
        'MlpPolicy',
        env=env, 
        learning_rate=7e-4,
        ent_coef=0.01,
        # n_steps=20,
        # batch_size=20,
        # gamma=0.99,
        tensorboard_log='log',
        verbose=0,
    )
    print(model.policy)
    
    name = name_generate(env, model, feature_extractor, tickers, postfix)
    try:
        model.learn(
            total_timesteps=50000,
            eval_env=None,
            eval_freq=0,
            n_eval_episodes=0,
        )
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        pass

    model.save(f'log/{name}')

    mean, std = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean:.2f} +/- {std: .2f}")