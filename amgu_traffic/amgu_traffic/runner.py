import glob
import numpy as np
from .enviorment import CFWrapper
from amgu_abstract import RunnerWrapper
import os
import shutil
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.dqn as dqn
import cv2
import ray
import imageio


torch, nn = try_import_torch()

__all__ = ["RayRunner"]


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
        script_dir = os.path.dirname(__file__)
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

    def eval(self, weight_path=None):
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
        dir_path = os.path.join(self.res_path, "Images")

        # remvove if file exist
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        idx = 0
        size = None
        while not done:
            action_np = agent_instance.compute_single_action(obs_np)
            obs_tensor = torch.from_numpy(obs_np)[None, :].float()
            obs_img = self._convert_to_image(obs_np)
            size = obs_img.shape if size == None else size
            cv2.imwrite(os.path.join(dir_path, f"{idx}.png"), obs_img)
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
            idx += 1
        assert size is not None
        self._save_gif(dir_path, size[:-1])

    def _convert_to_image(self, obs_np):
        assert type(obs_np) is np.ndarray
        intersection_num = obs_np.shape[1]
        new_shape = (
            obs_np.shape[0],
            intersection_num * obs_np.shape[2],
            intersection_num * obs_np.shape[3],
        )
        return np.reshape(obs_np, new_shape).T.astype(np.uint8)

    def _save_gif(self, path, frame_size, fps=1.0):
        images_path = glob.glob(f"{path}/*.png")
        with imageio.get_writer(f"{path}/movie.gif", mode="I") as writer:
            for filename in images_path:
                image = imageio.imread(filename)
                writer.append_data(image)

        # frame = cv2.imread(images_path[0])
        # height, width, layers = frame.shape
        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # video = cv2.VideoWriter('{path}/video.mp4', fourcc,fps, (width, height))
        # for filename in images_path:
        #     video.write(cv2.imread(filename))
        # cv2.destroyAllWindows()
        # video.release()
