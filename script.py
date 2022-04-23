import json
from ray import tune
from enviorments import SingleAgentCityFlow
from utils import AlgorithemsConfig,ModelConfig
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import argparse


# =============== CONST ===============
ALGORITHM_MAPPER = {
    "A3C": AlgorithemsConfig.A3C,
    "PPO":  AlgorithemsConfig.PPO,
    }
AGENT_MAPPER = {
    "A3C": a3c.A3CTrainer,
    "PPO": ppo.PPOTrainer,
    }
# =============== CLI ARGS ===============
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', action='store_true')
parser.set_defaults(evaluation=False)
# share arguments
parser.add_argument("--seed", type=int, default=123, help="RLLIB Seed.")
parser.add_argument("--framework", choices=["tf", "tf2", "tfe", "torch"], default="torch", help="Choose NN Framework.")
parser.add_argument("--evaluation-interval", type=int, default=3, help="How Many Train Evalation.")
parser.add_argument("--evaluation-episodes", type=int, default=20, help="How Many Evalation Episodes.")
parser.add_argument('--lr', nargs='+', type=int, default=[0.1], help="Which LR To Run Expirent MULTI.")
parser.add_argument("--config-path", type=str, default="examples/1x1/config.json", help="Path For Config File (cityflow).")
parser.add_argument("--steps-per-episode", type=int, default=1000, help="Number Of Step Before ENV Reset.")
parser.add_argument("--reward-function", choices=["waiting_count", "avg_travel_time", "delay_from_opt", "exp_delay_from_opt"],default="delay_from_opt", help="Choose Reward Function For Cityflow.")
parser.add_argument("--algorithm", choices=["A3C", "PPO"],default="PPO", help="Choose Algorithm From Ray.")
parser.add_argument("--result-path", type=str ,default="res/", help="Choose Path To Save Result.")
parser.add_argument("--max-timesteps", type=int ,default=10_000, help="Stop After max-timesteps Iterations.")
parser.add_argument("--load-from", type=str, help="Result Directory Path (trained).")
# =============== Script ===============
args = parser.parse_args()
register_env("CityFlows", lambda config: SingleAgentCityFlow(config))
config = ALGORITHM_MAPPER[args.algorithm]
config["env"]="CityFlows"
config["seed"] = args.seed
config["framework"] = args.framework
config["evaluation_interval" ]= args.evaluation_interval
config["evaluation_num_episodes"] = args.evaluation_episodes
config["env_config"]={
        "config_path": args.config_path,
        "steps_per_episode": args.steps_per_episode,
        "reward_func": args.reward_function,
    }
# PRINT INFORMATION TO USER
print("="*15," RUNNING WITH THE FOLLOWING CONFIG ","="*15)
print(json.dumps(config, indent=2, sort_keys=True))
print("With Actor:",args.algorithm)
print("="*65)
# DECIDE ON EVALUATION/TRAIN
if args.evaluation:
    agent = AGENT_MAPPER[args.algorithm](config=config)
    agent.restore(args.load_from)
    # Todo: add implementation of evaloition run x step in env and save info for the future
else:
    config["lr"]= tune.grid_search(args.lr)
    tune.run(args.algorithm,config=config,local_dir=args.result_path,checkpoint_at_end=True,mode="min",stop={"timesteps_total": args.max_timesteps})