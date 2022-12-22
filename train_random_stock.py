from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from torch.nn import Tanh
from stock_env.envs import *
from stock_env.feature.feature_extractor import *
from stock_env.envs import RandomStockEnv


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


if __name__ == "__main__":
    # path = "../stock_datasets/"
    # tickers = "SSI VND HCM MBS VCI".split()
    # n_steps = 5
    # suffix = "sp500"
    suffix = "maml"
    feature_extractor = TrendFeatures

    env = gym.make("SP500-v0")
    eval_env = gym.make("FinService-v0")

    # model = PPO(
    #     'MlpPolicy',
    #     env=env,
    #     learning_rate=4.6e-5,
    #     ent_coef=2.45e-5,
    #     n_steps=1024,
    #     batch_size=64,
    #     clip_range=0.1,
    #     n_epochs=10,
    #     gamma=0.95,
    #     max_grad_norm=0.3,
    #     vf_coef=0.615,
    #     tensorboard_log='log',
    #     verbose=1,
    #     policy_kwargs=dict(
    #         activation_fn=Tanh,
    #         net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    #     )
    # )
    # setup noise

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions)
    )
    model = SAC(
        "MlpPolicy",
        env=env,
        learning_rate=0.0009324565055470252,
        gamma=0.98,
        buffer_size=1000000,
        batch_size=1024,
        train_freq=(16, "step"),
        gradient_steps=16,
        learning_starts=50000,
        action_noise=action_noise,
        tau=0.2,
        use_sde=True,
        tensorboard_log="log",
        policy_kwargs=dict(log_std_init=-2.4865866564874546, net_arch=[64, 64]),
        verbose=1,
    )
    print(model.policy)

    name = name_generate(
        env, model, feature_extractor=feature_extractor, ticker=None, suffix=suffix
    )
    try:
        model.learn(
            total_timesteps=100000,
            eval_env=eval_env,
            eval_freq=50000,
            n_eval_episodes=10,
        )
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        pass

    model.save(f"log/{name}")

    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Mean reward: {mean:.2f} +/- {std: .2f}")
