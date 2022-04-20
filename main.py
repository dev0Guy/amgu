from ray import tune
from enviorments import SingleAgentCityFlow
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

register_env("CityFlows", lambda config: SingleAgentCityFlow(config))
config = {
    "env": "CityFlows",
    "env_config": {
        "config_path":'examples/1x1/config.json',
        "steps_per_episode": 1_000,
        "reward_func": "exp_delay_from_opt",
    },
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "framework":"torch",
    "evaluation_interval": 2,
    "evaluation_num_episodes": 20,
    "log_level": "DEBUG",
    "clip_rewards": True,
    "lr": tune.grid_search([0.1, 0.01, 0.001]),
    }

analysis = tune.run("PPO",config=config,
    stop={
        "timesteps_total": 10_000,
    },
    local_dir="res",checkpoint_at_end=True)
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")
# or simply get the last checkpoint (with highest "training_iteration")
last_checkpoint = analysis.get_last_checkpoint()
# if there are multiple trials, select a specific trial or automatically
# choose the best one according to a given metric
last_checkpoint = analysis.get_last_checkpoint(
    metric="episode_reward_mean", mode="max"
)