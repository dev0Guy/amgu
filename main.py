# =============== IMPORTS ===============
import numpy as np
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.dqn as dqn
from enviorments import *
import argparse
import models
import pickle
import torch
import json
import ray
from attacks import Attacks
# ray.init(log_to_driver=False)
# =============== CONST ===============
ALGORITHM_CONFIG_MAPPER = {
    "A3C": {},
    "PPO": {},
    "DQN": {
        # Discount factor of the MDP.
        "gamma": 0.99,
        # The default learning rate.
        "lr": 0.01,
    },
}
ALGORITHM_MAPPER = {
    "A3C": a3c.A3CTrainer,
    "PPO": ppo.PPOTrainer,
    "DQN": dqn.DQNTrainer,
}
ATTACK_MAPPER = {
    "WhiteTargeted": Attacks.whitebox_targeted,
    "WhiteUntargeted": Attacks.whitebox_untargeted,
    "BlackTargeted": Attacks.blackbox_targeted,
    "BlackUntargeted": Attacks.blackbox_untargeted,
}

# =============== CLI ARGS ===============
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', action='store_true', help="Evaluation run")
parser.add_argument("--multi-agent",action='store_true', help="Single Or Multi agent config")
parser.add_argument('--attack', action='store_true', help="Attack run")
parser.set_defaults(evaluation=False)
parser.set_defaults(multi_agent=False)
parser.set_defaults(attack=False)
# share arguments
parser.add_argument("--seed", type=int, default=123, help="RLLIB Seed.")
parser.add_argument("--framework", choices=["tf", "tf2", "tfe", "torch"], default="torch", help="Choose NN Framework.")
parser.add_argument("--evaluation-interval", type=int, default=3, help="How Many Train Evalation.")
parser.add_argument("--evaluation-duration", type=int, default=3, help="How Many Evalation Episodes.")
parser.add_argument('--lr', nargs='+', type=float, default=[0.9], help="Which LR To Run Expirent MULTI.")
parser.add_argument("--config-path", type=str, default="examples/hangzhou_1x1_bc-tyc_18041607_1h/config.json", help="Path For Config File (cityflow).")
parser.add_argument("--steps-per-episode", type=int, default=400, help="Number Of Step Before ENV Reset.")
parser.add_argument("--reward-function", choices=["waiting_count", "avg_travel_time", "delay_from_opt", "exp_delay_from_opt"],default="waiting_count", help="Choose Reward Function For Cityflow.")
parser.add_argument("--algorithm", choices=["A3C", "PPO","DQN"],default="DQN", help="Choose Algorithm From Ray.")
parser.add_argument("--result-path", type=str ,default="res/", help="Choose Path To Save Result.")
parser.add_argument("--max-iter", type=int ,default=100, help="Stop After max-timesteps Iterations.")
parser.add_argument("--load-from", type=str, help="Result Directory Path (trained).")
parser.add_argument("--model", choices=["CNN","Prototype","Queue"],default="CNN", help="Choose Model For Algorithm To Run.")
parser.add_argument("--attack-kind", choices=["WhiteTargeted","WhiteUntargeted","BlackTargeted","BlackUntargeted",''],default="", help="Choose Model For Algorithm To Run.")

args = parser.parse_args()
# =============== Register To RLlib ===============
# ENV
register_env("Single-CityFlow", lambda config: SingleIntersection(config))
# MODELS
ModelCatalog.register_custom_model("CNN", models.CNN)
ModelCatalog.register_custom_model("Prototype", models.Prototype)
ModelCatalog.register_custom_model("Queue", models.Queue)
# =============== CONFIG ===============
config = ALGORITHM_CONFIG_MAPPER[args.algorithm]
config["env"] = "Multi-CityFlow" if args.multi_agent else "Single-CityFlow"
config["seed"] = args.seed
config["framework"] = args.framework
config["evaluation_interval" ]= args.evaluation_interval
config["evaluation_duration"] = args.evaluation_duration
config['exploration_config'] = {
    'type': "EpsilonGreedy",
    'epsilon_schedule': {
        'type': "ExponentialSchedule",
        'initial_p': 1,
        'schedule_timesteps': args.steps_per_episode//5,
        'decay_rate': 0.99,
    }
}
    
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
# =============== RUN ===============
is_queue = config['model']['custom_model'] == "Queue"
if args.evaluation:
    # creating a cityflow env
    env = SingleIntersection(config["env_config"]) 
    # loading the old agent 
    agent = None if is_queue else ALGORITHM_MAPPER[args.algorithm](config=config)
    if agent != None:
        agent.restore(args.load_from)
    episodes_reward = []
    model = agent.get_policy().model._network if not is_queue else models.Queue(env.action_impact)
    _preprocess = agent.get_policy().model._preprocess
    res_dict = {'rewards': [], 'ATT': [], 'QL': []} # for each episode
    action_func = agent.compute_single_action if not is_queue else model
    def create_tensor(obs):
        return torch.from_numpy(obs[None,:]).float()
    results = []
    for episode_num in range(1):
        episode_reward = 0
        done = False
        obs = env.reset() # img
        tensor_obs = create_tensor(obs)
        x = 0
        while not done:
            action = action_func(obs) # labels
            if args.attack_kind != '' and not is_queue:
                action = np.array([int(action)]) if isinstance(action,np.int32) else action
                tensor_action = torch.zeros(*action.shape)
                attack_obs = ATTACK_MAPPER[args.attack_kind](model, tensor_obs, tensor_action,_preprocess) # adv_img
                attack_action = action_func(attack_obs)
                attack_action = tensor_action.numpy()
                attack_action = np.array([int(attack_action)]) if isinstance(attack_action,np.int32) else attack_action
                results.append({'obs': obs,'adv_obs': attack_obs, 'action': action, 'adv_action': attack_action, "obs_preprocess": _preprocess(tensor_obs)})
                action = attack_action
            obs, reward, done, info = env.step(action)
            tensor_obs = create_tensor(obs)
            episode_reward += reward
            x += 1
        # saving the rewards, ATT, QL from each episode
        res_dict['rewards'].append(episode_reward)
        res_info = env.get_results()
        res_dict['ATT'].append(res_info['ATT'])
        res_dict['QL'].append(res_info['QL'])
    path_ = args.load_from.split('/')
    file_name = args.attack_kind if args.attack_kind != '' else "vanila"
    with open(f"{args.result_path}/{file_name}_results.pkl","wb") as f:
        pickle.dump(results, f)
    with open(f"{args.result_path}/{file_name}.json","w") as file:
        json.dump(res_dict, file)
else:
    if is_queue:
        raise Warning("Can't Train With Queue Model Only Interface!")
    ray.tune.run(args.algorithm,config=config,local_dir=args.result_path,checkpoint_at_end=True,mode="min",
                    stop={"training_iteration": args.max_iter})
    ray.shutdown()