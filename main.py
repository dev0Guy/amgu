from ray import tune
from enviorments import SingleAgentCityFlow
from utils import AlgorithemsConfig,ModelConfig
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

register_env("CityFlows", lambda config: SingleAgentCityFlow(config))
config = AlgorithemsConfig.PPO
config["env"]="CityFlows"
config["seed"]=123
config["framework"]="torch"
config["evaluation_interval"]=3
config["evaluation_num_episodes"]=20
config["lr"]= tune.grid_search([0.1, 0.01, 0.001])
config["env_config"]={
        "config_path":'examples/1x1/config.json',
        "steps_per_episode": 1_500,
        "reward_func": "delay_from_opt",
    }
analysis = tune.run("PPO",config=config,local_dir="res",checkpoint_at_end=True,mode="min",
    stop={"timesteps_total": 110_000,})
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")
# or simply get the last checkpoint (with highest "training_iteration")
last_checkpoint = analysis.get_last_checkpoint()
# if there are multiple trials, select a specific trial or automatically
# choose the best one according to a given metric
last_checkpoint = analysis.get_last_checkpoint(
    metric="episode_reward_mean", mode="min"
)
print(last_checkpoint)