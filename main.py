from amgu_traffic import DiscreteCF, AvgWaitingTime, CNN, RayRunner, Vanila

agent_name = "DQN"


config = {
    "env_config": {
        "config_path": "examples/hangzhou_1x1_bc-tyc_18041607_1h/config.json",
        "steps_per_episode": 100,
        "res_path": f"res/res_{agent_name}/",
    },
    "stop": {"training_iteration": 5},
    "res_path": f"res/res_{agent_name}/",
    "framework": "torch",
    "seed": 123,
    "evaluation_interval": 10,
    "evaluation_duration": 5,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "epsilon_schedule": {
            "type": "ExponentialSchedule",
            "initial_p": 1,
            "schedule_timesteps": 100 // 5,
            "decay_rate": 0.99,
        },
    },
    "model": {
        "custom_model": "new_models",
        "custom_model_config": {
            "intersection_num": 1,
            "hidden_size": 10,
        },
    },
    "run_from": "/Users/guyarieli/Documents/GitHub/amgu/amgu/",
    "env": "custom_env",
}
preprocess_dict = {"func": Vanila, "argument_list": []}
env_func = lambda _: DiscreteCF(config["env_config"], AvgWaitingTime, preprocess_dict)
runner = RayRunner(config, CNN, env_func, agent_name)
runner.train()
runner.eval()
