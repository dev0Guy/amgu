import json
from ray import tune
from enviorments import SingleAgentCityFlow,MultiAgentCityFlow
from utils import AlgorithemsConfig
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import argparse
import ray
from attacks import Attacks
import torch
from ray.rllib.models import ModelCatalog
import models
# =============== CONST ===============
ALGORITHM_MAPPER = {
    "A3C": AlgorithemsConfig.A3C,
    "PPO":  AlgorithemsConfig.PPO,
    }
AGENT_MAPPER = {
    "A3C": a3c.A3CTrainer,
    "PPO": ppo.PPOTrainer,
    }

# MODEL_MAPPER = {
#     "FCN": ModelConfig.FCN,
# }
# =============== CLI ARGS ===============
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', action='store_true', help="Evaluation run")
parser.add_argument("--multi-agent",action='store_true', help="Single Or Multi agent config")
parser.set_defaults(evaluation=False)
parser.set_defaults(multi_agent=False)
# share arguments
parser.add_argument("--seed", type=int, default=123, help="RLLIB Seed.")
parser.add_argument("--framework", choices=["tf", "tf2", "tfe", "torch"], default="torch", help="Choose NN Framework.")
parser.add_argument("--evaluation-interval", type=int, default=3, help="How Many Train Evalation.")
parser.add_argument("--evaluation-episodes", type=int, default=20, help="How Many Evalation Episodes.")
parser.add_argument('--lr', nargs='+', type=float, default=[0.1], help="Which LR To Run Expirent MULTI.")
parser.add_argument("--config-path", type=str, default="examples/1x1/config.json", help="Path For Config File (cityflow).")
parser.add_argument("--steps-per-episode", type=int, default=800, help="Number Of Step Before ENV Reset.")
parser.add_argument("--reward-function", choices=["waiting_count", "avg_travel_time", "delay_from_opt", "exp_delay_from_opt"],default="waiting_count", help="Choose Reward Function For Cityflow.")
parser.add_argument("--algorithm", choices=["A3C", "PPO"],default="PPO", help="Choose Algorithm From Ray.")
parser.add_argument("--result-path", type=str ,default="res/", help="Choose Path To Save Result.")
parser.add_argument("--max-timesteps", type=int ,default=80_000, help="Stop After max-timesteps Iterations.")
parser.add_argument("--load-from", type=str, help="Result Directory Path (trained).")
parser.add_argument("--model", choices=["CNN","Prototype"],default="Prototype", help="Choose Model For Algorithm To Run.")
args = parser.parse_args()
# ===============  ===============
# runtime_env = {"working_dir": "./"}
# ray.init(runtime_env=runtime_env)
ray.init(log_to_driver=False)
# =============== Register To RLlib ===============
# tegister envs
register_env("Single-CityFlow", lambda config: SingleAgentCityFlow(config))
register_env("Multi-CityFlow", lambda config: MultiAgentCityFlow(config))
# register model
ModelCatalog.register_custom_model("CNN", models.CNN)
ModelCatalog.register_custom_model("FCN", models.FCN)
ModelCatalog.register_custom_model("Prototype", models.Prototype)
# =============== CONFIG ===============
config = ALGORITHM_MAPPER[args.algorithm]
config["env"] = "Multi-CityFlow" if args.multi_agent else "Single-CityFlow"
config["seed"] = args.seed
config["framework"] = args.framework
config["evaluation_interval" ]= args.evaluation_interval
config["evaluation_num_episodes"] = args.evaluation_episodes
config["env_config"]={
        "config_path": args.config_path,
        "steps_per_episode": args.steps_per_episode,
        "reward_func": args.reward_function,
    }
config["model"] = {
            "custom_model": args.model,
        }
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
    # agent.restore(args.load_from)
    # creating a cityflow env
    env = SingleAgentCityFlow(config["env_config"]) 
    episodes_reward = []
    # model = agent.get_policy().model.logits_layer
    model = agent.get_policy().model._network
    for episode_num in range(args.evaluation_episodes):
        episode_reward = 0
        done = False
        obs = env.reset() 
        for step in range(args.steps_per_episode):
            tensor_obs = torch.from_numpy(obs)[None,:].float()
            action = agent.compute_single_action(obs)
            tensor_action = torch.from_numpy(action) 
            attack_obs = Attacks.GN(model, tensor_obs, tensor_action) 
            action_ = model(attack_obs)
            action_ = torch.reshape(action_,(len(action),-1))
            action_ = torch.argmax(action_, dim=1)
            obs, reward, done, info = env.step(action_.numpy())
            # episode_reward += reward
        # saving the rewards from each episode
        episodes_reward.append(episode_reward)
else:
    config["lr"]= tune.grid_search(args.lr)
    tune.run(args.algorithm,config=config,local_dir=args.result_path,checkpoint_at_end=True,mode="min",stop={"timesteps_total": args.max_timesteps})