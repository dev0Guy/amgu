from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from functools import cmp_to_key
import numpy as np
import glob
import gym
import os


def stable_baseline_train(policy_class, env_class, config):

    assert issubclass(policy_class, BaseAlgorithm)
    assert issubclass(env_class, gym.Env)
    assert "env_config" in config
    assert "env_param" in config
    assert type(config["env_param"]) is dict
    assert (
        type(config["env_config"]) is dict
        and "steps_per_episode" in config["env_config"]
    )
    assert "policy_param" in config and "policy" in config["policy_param"]

    evaluation_interval = config.get("evaluation_interval", 20)
    evaluation_duration = config.get("evaluation_duration", 1)
    # lr_start = config['policy_param'].get('lr',0.05)

    # if 'lr' in config['policy_param']:
    #     del config['policy_param']['lr']

    iter_num = config.get("stop", {"training_iteration": 10}).get(
        "training_iteration", 10
    )

    env = env_class(config["env_config"], **config["env_param"])
    check_env(env, warn=False)

    train_step_number = config["env_config"].get("steps_per_episode", 100)
    total_timesteps = train_step_number * iter_num
    model = policy_class(env=env, **config["policy_param"])

    checkpoint_callback = CheckpointCallback(
        save_freq=evaluation_interval * train_step_number,
        save_path=config["experiment_name"],
        name_prefix="rl_model",
    )
    model.learn(total_timesteps, callback=checkpoint_callback)

    model.save(f"{config['experiment_name']}/{config['experiment_name']}")
