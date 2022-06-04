import numpy as np
from .environment import CFWrapper
from amgu_abstract import RunnerWrapper
from .visualization import *
import os
import shutil
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.dqn as dqn
import ray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

torch, nn = try_import_torch()

__all__ = ["RayRunner"]
WINDOW_SIZE = 50


def normlize(x):
    min_val = x.min(-1)[0].min(-1)[0]
    max_val = x.max(-1)[0].max(-1)[0]
    return (x - min_val[:, :, None, None]) / (
        max_val[:, :, None, None] - min_val[:, :, None, None]
    )


class RayRunner(RunnerWrapper):

    _options = ["DQN", "PPO", "A3C"]
    _instance = {"A3C": a3c.A2CTrainer, "PPO": ppo.PPOTrainer, "DQN": dqn.DQNTrainer}

    def __init__(self, config: dict, model, env_func: CFWrapper, agent: str):
        super().__init__(config, model, None, agent)
        assert type(agent) is str
        assert agent in self._options
        assert "res_path" in config
        assert "stop" in config
        assert "model" in config and "custom_model" in config["model"]
        assert "run_from" in config
        assert "env" in config
        self.stop = self.config["stop"]
        self.run_folder = config["run_from"]
        self.res_path = self.config["res_path"]
        self.env_func = env_func
        del self.config["stop"]
        del self.config["res_path"]
        del self.config["run_from"]
        org_path = self.config["env_config"]["config_path"]
        ray.init(log_to_driver=False)
        self.config["env_config"]["config_path"] = os.path.join(
            self.run_folder, org_path
        )
        self.config["env_config"]["res_path"] = os.path.join(
            self.run_folder, self.res_path
        )
        register_env(config["env"], self.env_func)
        ModelCatalog.register_custom_model(
            self.config["model"]["custom_model"], self.model
        )

    def train(self, attack=None, kind="min"):
        assert kind in ["min", "max"]
        self._analysis = ray.tune.run(
            self.agent,
            config=self.config,
            local_dir=self.res_path,
            checkpoint_at_end=True,
            mode=kind,
            stop=self.stop,
        )

        self.last_checkpoint = self._analysis.get_last_checkpoint()

    def eval(self, weight_path=None, attack_func=None, attack_name=None):
        first = weight_path != None and type(weight_path) is str
        second = weight_path == None and self.last_checkpoint != None
        assert first or second
        if second:
            weight_path = self.last_checkpoint
        agent_instance = self._instance[self.agent](config=self.config)
        agent_instance.restore(weight_path)
        env = self.env_func(self.config["env"])
        done = False
        obs_np = env.reset()

        information_dict = {"rewards": [], "ATT": [], "QL": []}
        dir_path = os.path.join(self.res_path, "Images_vanila")
        if attack_name != None:
            dir_path = os.path.join(self.res_path, f"Images_{attack_name}")

        # remvove if file exist
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        idx = 0
        size = None
        model = agent_instance.get_policy().model.network

        plt.figure(figsize=(10, 13), dpi=80)
        while not done:
            obs_tensor = torch.from_numpy(obs_np).float()
            if attack_func != None:
                obs_tensor = attack_func(
                    model, obs_tensor, torch.Tensor([[0]]), env.preprocess
                )
                obs_np = obs_tensor.numpy()
            action_np = agent_instance.compute_single_action(obs_np)
            obs_img = VisualizationCF.convert_to_image(obs_np)
            size = obs_img.shape if size == None else size
            if action_np is np.array:
                action_tensor = torch.reshape(action_np, (len(action_np), -1))
                action_tensor = torch.argmax(action_tensor, dim=1)
                action = action_tensor.numpy()
            else:
                action = action_np
            obs_np, reward, done, _ = env.step(action)
            information_dict["rewards"].append(reward)
            res_info = env.get_results()
            information_dict["ATT"].append(res_info["ATT"])
            information_dict["QL"].append(res_info["QL"])

            gs = gridspec.GridSpec(4, 3)
            gs.update(wspace=0.5)
            ax1 = plt.subplot(
                gs[:3, :3],
            )
            ax2 = plt.subplot(gs[3, 0])
            ax3 = plt.subplot(gs[3, 1])
            ax4 = plt.subplot(gs[3, 2])
            _from = idx - WINDOW_SIZE if idx >= WINDOW_SIZE else 0
            _to = len(information_dict["ATT"])
            x_axis = [i for i in range(_from, _to)]
            ax1.imshow(obs_img)
            ax2.plot(x_axis, information_dict["ATT"][_from:], color="forestgreen")
            ax2.set_title("ATT")
            ax3.bar(
                [i for i in range(len(res_info["QL"]))], res_info["QL"], color="coral"
            )
            ax3.set_title("QL")
            ax4.plot(x_axis, information_dict["rewards"][_from:], color="teal")
            ax4.set_title("Rewards")
            plt.savefig(os.path.join(dir_path, f"{idx}.png"))
            idx += 1
        assert size is not None
        VisualizationCF.save_gif(dir_path, size[:-1])
        with open(f'{dir_path}/information.json', 'w') as file:
            json.dump(information_dict, file)
