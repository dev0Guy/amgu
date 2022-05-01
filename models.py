from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import gym
import numpy as np
# import what installed
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

# ======= MODELS =======
class FCN(TorchModelV2, nn.Module):
    """
        Simple Fully Connected Network Model
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)        
        
        self.obs_type: type = type(obs_space)
        self.action_type: type = type(action_space)
        
        assert self.obs_type in [gym.spaces.box.Box,gym.spaces.Dict]
        assert self.action_type in [gym.spaces.MultiDiscrete,gym.spaces.Dict]
        
        input_size: int = np.prod(obs_space.shape)
        hidden_size: int = 100
        output_size: int = np.prod(action_space.shape) * num_outputs
        self._network: nn.Sequential = nn.Sequential(
            nn.Flatten(1,-1),
            nn.Linear(in_features=input_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=output_size),
        )
        self._output: torch.Tensor = None
        
        
    @override(ModelV2)
    def forward(self, input_dict: dict, state: list, seq_lens) -> tuple[torch.Tensor,list]:
        self._output = self._network(input_dict["obs"])
        return self._output, []

    @override(ModelV2)
    def value_function(self) -> torch.Tensor:
        assert self._output is not None, "must call forward first!"
        return torch.mean(self._output, -1)
    


