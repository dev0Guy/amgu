from ray import tune
from enviorments import SingleAgentCityFlow
from ray.tune.registry import register_env

register_env("CityFlows", lambda config: SingleAgentCityFlow(config))
tune.run("PPO",config={
    "env": "CityFlows",
    "env_config": {
        "config_path":'examples/2x3/config.json',
        "steps_per_episode": 1_000,
        "reward_func": "avg_travel_time",
    },
    "framework":"torch",
    "evaluation_interval": 2,
    "evaluation_num_episodes": 20,
    "log_level": "DEBUG",
},local_dir="city_v1")