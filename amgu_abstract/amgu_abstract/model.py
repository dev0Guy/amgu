from abc import abstractmethod
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override
import gym

# Try Import
torch, nn = try_import_torch()

__all__ = ["TorchWrapper", "AlgorithmWrapper"]

""" Model class,
    Define All Parent Model That Can Be Hindranced From.
    Define The API Of SubModules.
"""


class TorchWrapper(TorchModelV2, nn.Module):
    """ 
            Wrapper, Around All Torch Models (weighted models).
            Build On Top Of Model API, Easy To Gindrance & Expand.
        """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Constructors

            Args:
                obs_space (gym.spaces): Space Of Observations.
                action_space (gym.spaces): Space Of Actions.
                num_outputs (int): How Many Outputs The Model Will Return.
                model_config (dict): All Information To Build Model From.
                name (str): Name Of Model To Acess.
            """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.model_config = model_config["custom_model_config"]

        assert type(obs_space) in [gym.spaces.box.Box, gym.spaces.Dict]
        assert type(action_space) in [
            gym.spaces.MultiDiscrete,
            gym.spaces.Discrete,
            gym.spaces.Dict,
        ]

        self.obs_shape = obs_space.shape
        self.action_space = action_space.shape

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Call on object, activate the model (a.k.a network).
            Args:
                input_dict (dict): dict with all inforamtion from env.
                state (list):  list of last n state from env.
                seq_lens (_type_): the length of state list.
            Returns:
                torch.Tensor: the state after activated by the network.
            """
        input = input_dict["obs"].float()
        assert self.network is not None
        self._output = self.network(input)
        return self._output, []

    @override(TorchModelV2)
    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.mean(self._output, -1)


class AlgorithmWrapper(object):
    """ 
        Wrapper, Around All Algorithem Models (No Weighted).
        Build On Top Of Model API, Easy To Gindrance & Expand.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Constructors

            Args:
                obs_space (gym.spaces): Space Of Observations.
                action_space (gym.spaces): Space Of Actions.
                num_outputs (int): How Many Outputs The Model Will Return.
                model_config (dict): All Information To Build Model From.
                name (str): Name Of Model To Acess.
        """
        assert type(obs_space) in [gym.spaces.box.Box, gym.spaces.Dict]
        assert type(action_space) in [
            gym.spaces.MultiDiscrete,
            gym.spaces.Discrete,
            gym.spaces.Dict,
        ]

        self.obs_shape = obs_space.shape
        self.action_space = action_space.shape
        self.model_config = model_config["custom_model_config"]

    @abstractmethod
    def __call__(self, obs: torch.Tensor):
        """Call on object, activate Model.
        Args:
            input_dict (dict): dict with all inforamtion from env.
            state (list):  list of last n state from env.
            seq_lens (_type_): the length of state list.
        Returns:
            torch.Tensor: the state after activated by the network.
        """
        return obs
