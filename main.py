from ray import tune
from enviorments import SingleAgentCityFlow
from ray.tune.registry import register_env

register_env("CityFlows", lambda config: SingleAgentCityFlow(config))
tune.run("PPO",config={
    "env": "CityFlows",
    "env_config": {
        "config_path":'examples/1x1/config.json',
        "steps_per_episode": 1_000,
        "reward_func": "delay_from_opt",
    },
    "framework":"torch",
    "evaluation_interval": 2,
    "evaluation_num_episodes": 20,
    "log_level": "DEBUG"
    },
    stop={
        "timesteps_total": 100_000,
    },
    local_dir="res",
    sync_config= tune.SyncConfig(
        upload_dir="s3://bucket-name/sub-path/"
    ),
    )