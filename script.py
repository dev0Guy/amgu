import json
from ray import tune
from enviorments import SingleAgentCityFlow,MultiAgentCityFlow
from utils import AlgorithemsConfig
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import argparse
import ray
# from attacks import Attacks
import torch
from ray.rllib.models import ModelCatalog
import models
import pickle
import numpy as np

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
parser.add_argument('--attack', action='store_true', help="Attack run")
parser.add_argument('--evaluation', action='store_true', help="Evaluation run")
parser.add_argument("--multi-agent",action='store_true', help="Single Or Multi agent config")
parser.set_defaults(attack=False)
parser.set_defaults(evaluation=False)
parser.set_defaults(multi_agent=False)
# share arguments
parser.add_argument("--seed", type=int, default=123, help="RLLIB Seed.")
parser.add_argument("--framework", choices=["tf", "tf2", "tfe", "torch"], default="torch", help="Choose NN Framework.")
parser.add_argument("--evaluation-interval", type=int, default=4, help="How Many Train Evalation.")
parser.add_argument("--evaluation-duration", type=int, default=1, help="How Many Evalation Episodes.")
parser.add_argument('--lr', nargs='+', type=float, default=[0.9], help="Which LR To Run Expirent MULTI.")
parser.add_argument("--config-path", type=str, default="examples/hangzhou_1x1_bc-tyc_18041607_1h/config.json", help="Path For Config File (cityflow).")
parser.add_argument("--steps-per-episode", type=int, default=400, help="Number Of Step Before ENV Reset.")
parser.add_argument("--reward-function", choices=["waiting_count", "avg_travel_time", "delay_from_opt", "exp_delay_from_opt"],default="waiting_count", help="Choose Reward Function For Cityflow.")
parser.add_argument("--algorithm", choices=["A3C", "PPO"],default="PPO", help="Choose Algorithm From Ray.")
parser.add_argument("--result-path", type=str ,default="res/", help="Choose Path To Save Result.")
parser.add_argument("--max-iter", type=int ,default=20, help="Stop After max-timesteps Iterations.")
parser.add_argument("--load-from", type=str, help="Result Directory Path (trained).")
parser.add_argument("--model", choices=["CNN","Prototype","Queue"],default="Prototype", help="Choose Model For Algorithm To Run.")
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
ModelCatalog.register_custom_model("Queue", models.Queue)
ModelCatalog.register_custom_model("Prototype", models.Prototype)
# =============== CONFIG ===============
config = ALGORITHM_MAPPER[args.algorithm]
config["env"] = "Multi-CityFlow" if args.multi_agent else "Single-CityFlow"
config["seed"] = args.seed
config["framework"] = args.framework
config["env_config"]={
        "config_path": args.config_path,
        "steps_per_episode": args.steps_per_episode,
        "reward_func": args.reward_function,
    }
config['evaluation_interval'] = args.evaluation_interval
config['evaluation_duration'] = args.evaluation_duration
config["model"] = {
            "custom_model": args.model,
        }
# PRINT INFORMATION TO USER
print("="*15," RUNNING WITH THE FOLLOWING CONFIG ","="*15)
print(json.dumps(config, indent=2, sort_keys=True))
print("With Actor:",args.algorithm)
print("With Model:",args.model)
print("="*65)
is_queue = config['model']['custom_model'] == "Queue"
# DECIDE ON EVALUATION/TRAIN
if args.evaluation:
    # creating a cityflow env
    env = SingleAgentCityFlow(config["env_config"]) 
    # loading the old agent 
    agent = AGENT_MAPPER[args.algorithm](config=config)
    agent.restore(args.load_from)
    episodes_reward = []
    # model = agent.get_policy().model.logits_layer
    model = agent.get_policy().model._network if not is_queue else models.Queue(env.action_impact)
    res_dict = {'rewards': [], 'ATT': [], 'QL': []} # for each episode
    for episode_num in range(args.evaluation_episodes):
        episode_reward = 0
        done = False
        obs = env.reset() 
        for step in range(args.steps_per_episode):
            action = agent.compute_single_action(obs) if not is_queue else model(obs)
            if args.attack and not is_queue:
                tensor_obs = torch.from_numpy(obs)[None,:].float()
                tensor_action = torch.from_numpy(action) 
                # attack_obs = Attacks.GN(model, tensor_obs, tensor_action) 
                # action_ = model(attack_obs)
                action_ = action
                action_ = torch.reshape(action_,(len(action),-1))
                action_ = torch.argmax(action_, dim=1)
                obs, reward, done, info = env.step(action_.numpy())
            else:
                obs, reward, done, info = env.step(action)
            episode_reward += reward
        # saving the rewards, ATT, QL from each episode
        res_dict['rewards'].append(episode_reward)
        res_info = env.get_results()
        res_dict['ATT'].append(res_info['ATT'])
        res_dict['QL'].append(res_info['QL'])
    path_ = args.load_from.split('/')
    name_result = path_[3] + '_' + path_[5]
    if args.attack:
        attack_name = 'GN'
        with open(f"{args.result_path}/attacks/{name_result}.pkl","wb") as file:
            pickle.dump(res_dict, file)
    else:
        with open(f"{args.result_path}/{name_result}.pkl","wb") as file:
            pickle.dump(res_dict, file)
else:
    if is_queue:
        raise Warning("Can't Train With Queue Model Only Interface!")
    # config["lr"]= tune.grid_search(args.lr)
    tune.run(args.algorithm,config=config,local_dir=args.result_path,checkpoint_at_end=True,mode="min",
                stop={
                    "training_iteration": args.max_iter,
                }
            )
    ray.shutdown()