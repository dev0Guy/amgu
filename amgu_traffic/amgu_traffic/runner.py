from .enviorment import CFWrapper
from amgu_abstract import RunnerWrapper
import os
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog
import ray

torch, nn = try_import_torch()

__all__ = ["RayRunner"]


class RayRunner(RunnerWrapper):
    def __init__(self, config: dict, model, env_func: CFWrapper, agent: str):
        # env = env_func(config['env_config'])
        super().__init__(config, model, None, agent)
        assert type(agent) is str
        assert "res_path" in config
        assert "stop" in config
        assert "model" in config and "custom_model" in config["model"]
        assert "run_from" in config
        assert "env" in config
        self.stop = self.config["stop"]
        self.run_folder = config["run_from"]
        self.res_path = self.config["res_path"]
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
        register_env(config["env"], env_func)
        ModelCatalog.register_custom_model(
            self.config["model"]["custom_model"], self.model
        )

    def train(self, attack=None):
        ray.tune.run(
            self.agent,
            config=self.config,
            local_dir=self.res_path,
            checkpoint_at_end=True,
            mode="min",
            stop=self.stop,
        )

    def eval(self, attack=None):
        pass
