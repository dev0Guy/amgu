from Amgu.basic.env import CityFlow1D
from Amgu.basic.reward import *
from Amgu.basic.models import Random
from Amgu.basic.evaluate import evaluation_generator
from Amgu.visualization.graph import line_graph
from Amgu.runnner import stable_baseline_train
from stable_baselines3 import DQN

exp_name = "DQN_delta_waiting_time_1x1"
stable_baselines_config = {
    "experiment_name": exp_name,
    "env_config": {
        "config_path": "example/1x1/config.json",
        "steps_per_episode": 400,
        "save_path": "example/1x1/res/",
    },
    "env_param": {"reward_func": queue_length, "district": True},
    "policy_param": {
        "policy": "MlpPolicy",
        "tensorboard_log": f"{exp_name}/tesnorboard",
        # 'policy_kwargs': dict(activation_fn=th.nn.ReLU, net_arch=[146,50,8]),
        "gamma": 0.95,
        "learning_rate": 0.005,
        # "exploration_initial_eps": 1,
        # 'exploration_fraction': 0.9,
        # "exploration_final_eps": 0.15,
        # 'target_update_interval': 1_000,
    },
    "evaluation_interval": 400,
    "evaluation_duration": 1,
    "stop": {"training_iteration": 2_000},
}
stable_baseline_train(DQN, CityFlow1D, stable_baselines_config)
# env = CityFlow1D(
#     stable_baselines_config["env_config"], **stable_baselines_config["env_param"]
# )

# models = [
#     Random(env.action_space),
#     Random(env.action_space),
#     Random(env.action_space),
#     Random(env.action_space),
# ]

# gen = evaluation_generator(CityFlow1D, stable_baselines_config, models)
# line_graph(gen, 400, len(models), "here.png")
