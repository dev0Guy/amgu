import json
from ray import tune
from enviorments import SingleAgentCityFlow
from utils import AlgorithemsConfig,ModelConfig
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import argparse
from attacks import Attacks
import torch

# =============== CONST ===============
ALGORITHM_MAPPER = {
    "A3C": AlgorithemsConfig.A3C,
    "PPO":  AlgorithemsConfig.PPO,
    }
AGENT_MAPPER = {
    "A3C": a3c.A3CTrainer,
    "PPO": ppo.PPOTrainer,
    }
MODEL_MAPPER = {
    "FCN": ModelConfig.FCN,
    }
# =============== CLI ARGS ===============
parser = argparse.ArgumentParser()
parser.set_defaults(evaluation=False)
parser.add_argument('--evaluation', action='store_true')
# share arguments
parser.add_argument("--seed", type=int, default=123, help="RLLIB Seed.")
parser.add_argument("--framework", choices=["tf", "tf2", "tfe", "torch"], default="torch", help="Choose NN Framework.")
parser.add_argument("--evaluation-interval", type=int, default=3, help="How Many Train Evalation.")
parser.add_argument("--evaluation-episodes", type=int, default=20, help="How Many Evalation Episodes.")
parser.add_argument('--lr', nargs='+', type=float, default=[0.1], help="Which LR To Run Expirent MULTI.")
parser.add_argument("--config-path", type=str, default="examples/1x1/config.json", help="Path For Config File (cityflow).")
parser.add_argument("--steps-per-episode", type=int, default=1000, help="Number Of Step Before ENV Reset.")
parser.add_argument("--reward-function", choices=["waiting_count", "avg_travel_time", "delay_from_opt", "exp_delay_from_opt"],default="delay_from_opt", help="Choose Reward Function For Cityflow.")
parser.add_argument("--algorithm", choices=["A3C", "PPO"],default="PPO", help="Choose Algorithm From Ray.")
parser.add_argument("--result-path", type=str ,default="res/", help="Choose Path To Save Result.")
parser.add_argument("--max-timesteps", type=int ,default=10_000, help="Stop After max-timesteps Iterations.")
parser.add_argument("--load-from", type=str, help="Result Directory Path (trained).")
parser.add_argument("--model", choices=["FCN"],default="FCN", help="Choose Model For Algorithm To Run.")
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
config["model"]= MODEL_MAPPER[args.model]

# PRINT INFORMATION TO USER
print("="*15," RUNNING WITH THE FOLLOWING CONFIG ","="*15)
print(json.dumps(config, indent=2, sort_keys=True))
print("With Actor:",args.algorithm)
print("With Model:",args.model)
print("="*65)
# DECIDE ON EVALUATION/TRAIN
if args.evaluation:
    # loading the old agent 
    agent = AGENT_MAPPER[args.algorithm](config=config)
    agent.restore(args.load_from)
    # creating a cityflow env
    env = SingleAgentCityFlow(config["env_config"]) 
    episodes_reward = []
    model = agent.get_policy().model.logits_layer
    for episode_num in range(args.evaluation_episodes):
        episode_reward = 0
        done = False
        obs = env.reset() 
        tensor_obs = torch.from_numpy(obs) 
        for step in range(args.steps_per_episode):
            action = agent.compute_single_action(obs)
            tensor_action = torch.from_numpy(action) # 
            print(f'model: {model} \n tensor action: {tensor_action}')
            attack_obs = Attacks.GN(model, tensor_obs, tensor_action) #
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            tensor_obs = torch.from_numpy(obs) #
        # saving the rewards from each episode
        episodes_reward.append(episode_reward)
else:
    config["lr"]= tune.grid_search(args.lr)
    tune.run(args.algorithm,config=config,local_dir=args.result_path,checkpoint_at_end=True,mode="min",stop={"timesteps_total": args.max_timesteps})