
from enviorments import SingleAgentCityFlow,SingleIntersection
from utils import AlgorithemsConfig,ModelConfig
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.dqn as dqn

from ray import tune
from ray.rllib.models import ModelCatalog
import models
import argparse
import json
import torch

# =============== CONST ===============
ALGORITHM_MAPPER = {
    "A3C": AlgorithemsConfig.A3C,
    "PPO":  AlgorithemsConfig.PPO,
    "DQN":  {},
    }
# =============== CLI ARGS ===============
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', action='store_true', help="Evaluation run")
parser.add_argument("--multi-agent",action='store_true', help="Single Or Multi agent config")
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
parser.add_argument("--algorithm", choices=["A3C", "PPO","DQN"],default="DQN", help="Choose Algorithm From Ray.")
parser.add_argument("--result-path", type=str ,default="res/", help="Choose Path To Save Result.")
parser.add_argument("--max-iter", type=int ,default=140, help="Stop After max-timesteps Iterations.")
parser.add_argument("--load-from", type=str, help="Result Directory Path (trained).")
parser.add_argument("--model", choices=["CNN","Prototype","Queue"],default="Prototype", help="Choose Model For Algorithm To Run.")
args = parser.parse_args()
# =============== Register To RLlib ===============
# tegister envs
register_env("Single-CityFlow", lambda config: SingleIntersection(config))
# register_env("Multi-CityFlow", lambda config: MultiAgentCityFlow(config))
# register model
ModelCatalog.register_custom_model("CNN", models.CNN)
ModelCatalog.register_custom_model("FCN", models.FCN)
ModelCatalog.register_custom_model("Prototype", models.Prototype)
ModelCatalog.register_custom_model("Queue", models.Queue)
# =============== CONFIG ===============
config = ALGORITHM_MAPPER[args.algorithm]
config["env"] = "Multi-CityFlow" if args.multi_agent else "Single-CityFlow"
config["seed"] = args.seed
config["framework"] = args.framework
config["evaluation_interval" ]= args.evaluation_interval
# config["evaluation_num_episodes"] = args.evaluation_episodes
config["env_config"] = {
        "config_path": args.config_path,
        "steps_per_episode": args.steps_per_episode,
        "reward_func": args.reward_function,
    }
config["model"] = {
    "custom_model": args.model,
}
# =============== Print In CLI ===============
print("="*15," RUNNING WITH THE FOLLOWING CONFIG ","="*15)
print(json.dumps(config, indent=2, sort_keys=True))
print("With Actor:",args.algorithm)
print("With Model:",args.model)
print("With ENV:",config["env"])
print("="*65)

tune.run(args.algorithm,config=config,local_dir=args.result_path,checkpoint_at_end=True,mode="min",
                stop={
                    "training_iteration": args.max_iter,
                }
            )
ray.shutdown()
# =============== Train | Evaluation ===============
# agent = ppo.PPOTrainer(config)
# dqn
# # policy = agent.get_policy()
# # creating a cityflow env
# env = SingleIntersection(config["env_config"]) 
# episodes_reward = []
# # model = policy.model
# model = models.Queue(env.action_impact)
# obs =  env.reset() 

# # for episode_num in range(args.evaluation_episodes):
# #     done = False
# for x in range(10):
#     obs = env.reset() 
#     episode_reward = 0
#     for step in range(args.steps_per_episode):
#         # obs_torch = torch.from_numpy(obs)[None,:].float()
#         action = agent.compute_single_action(obs)
#         obs, reward, done, info = env.step(action)
#         episode_reward += reward
#         # break
#     print(episode_reward)
#     #     # saving the rewards from each episode
#     #     episodes_reward.append(episode_reward)

# # print(policy.model) # Prints the model summary
# # action = agent.compute_single_action(obs)