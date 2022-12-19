from stock_env.exp_manager import ExperimentManager

if __name__ == "__main__":
    ARGS_PATH = "configs/maml.yaml"
    ENV_ID = "VNALL-v0"
    METHODS_STATE_DICT = {
        "random": None,
        "maml": "model/maml_sp500_20221218_112927.pth",
    }
    exp = ExperimentManager(
        args_path=ARGS_PATH, env_id=ENV_ID, methods_state_dict=METHODS_STATE_DICT
    )

    adaption_results = exp.mass_adaption_results(
        methods=["maml"],
        maybe_num_tasks=["SSI"],
        total_adapt_steps=5,
        n_eval_episodes=5,
    )

    print(adaption_results)

    trading_performance, _ = exp.mass_trading_performance(
        methods=["maml"],
        maybe_num_tasks=["SSI"],
        total_adapt_steps=5,
    )

    print(trading_performance)
