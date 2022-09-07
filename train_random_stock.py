from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stock_env.envs import *
from stock_env.feature.feature_extractor import *
import mt4_hst
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_linear_fn

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
    
    feature_extractor = TrendFeatures()
    env = RandomStockEnv(
        tickers = tickers,
        data_folder_path = path,
        feature_extractor = feature_extractor
    )
    
    policy_kwargs = dict(net_arch=[64, 128, 128])
    model = SAC(
        'MlpPolicy',
        env=env,
        learning_rate=7e-5,
        gamma=0.99,
        buffer_size=100000,
        batch_size=128,
        train_freq=(4, "step"),
        gradient_steps=1,
        learning_starts=5000,
        target_update_interval=1000,
        tensorboard_log='log',
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    print(model.policy)
    
    name = name_generate(env, model, feature_extractor, tickers, postfix)
    try:
        model.learn(
            total_timesteps=500000,
            eval_env=None,
            eval_freq=0,
            n_eval_episodes=0,
        )
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        pass

    model.save(f'log/{name}')

    mean, std = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Mean reward: {mean:.2f} +/- {std: .2f}")

    # run model to get detailed information in the enviroment
    done = False
    obs = env.reset(eval_mode=True)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

    # get data
    env.get_history().to_csv(f'temp/history/{name}.csv', index=False)