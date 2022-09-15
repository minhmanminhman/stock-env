from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch.nn import Tanh
from stock_env.envs import *
from stock_env.feature import feature_extractor
from stock_env.feature.feature_extractor import *
from stock_env.wrappers import StackObs
from stock_env.envs import RandomStockEnv
from stock_env.data_loader import RandomStockLoader

def name_generate(env, algo, feature_extractor, ticker, suffix=None):
    env_name = env.__class__.__name__
    algo_name = algo.__class__.__name__
    extractor_name = feature_extractor.__name__
    name = f"{algo_name}_{env_name}_{extractor_name}"
    if type(ticker) == str:
        name += f"_{ticker}"
    if suffix is not None:
        name += f"_{suffix}"
    return name

if __name__ == '__main__':
    path = "../stock_datasets/"
    tickers = "SSI VND HCM MBS VCI".split()
    suffix = "finservice"
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
        env="FinService-v0", 
        learning_rate=4.6e-5,
        ent_coef=2.45e-5,
        n_steps=1024,
        batch_size=64,
        clip_range=0.1,
        n_epochs=10,
        gamma=0.95,
        max_grad_norm=0.3,
        vf_coef=0.615,
        tensorboard_log='log',
        verbose=0,
        policy_kwargs=dict(
            activation_fn=Tanh,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        )
    )
    print(model.policy)
    
    name = name_generate(env, model, feature_extractor, tickers, suffix)
    try:
        model.learn(
            total_timesteps=100000,
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