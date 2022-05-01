from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

import gym
import numpy as np
# import what installed
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
# ======= MODELS =======
class _BaseModel(TorchModelV2, nn.Module):
    """
        Simple Fully Connected Network Model
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)        
        
        obs_type: type = type(obs_space)
        action_type: type = type(action_space)
        
        assert obs_type in [gym.spaces.box.Box,gym.spaces.Dict]
        assert action_type in [gym.spaces.MultiDiscrete,gym.spaces.Dict]
        
        self.multi = action_type is gym.spaces.Dict
        self._output: torch.Tensor = None  
        
    @override(ModelV2)
    def forward(self, input_dict: dict, state: list, seq_lens) -> tuple[torch.Tensor,list]: 
        input = input_dict["obs"]
        if type(input) is torch.Tensor:
            self._output = self._network(input)
        else:
            tmp_lst = []
            for x in input.values():
                tmp_lst.append(self._network(x))
            self._output = torch.permute(torch.stack(tmp_lst),(1,0,2)).flatten(1,-1)
        return self._output, []

    @override(ModelV2)
    def value_function(self) -> torch.Tensor:
        assert self._output is not None, "must call forward first!"
        return torch.mean(self._output, -1)

class FCN(_BaseModel):
    """
        Simple Fully Connected Network Model
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs,model_config, name)
        input_size: int = np.prod(obs_space.shape)
        if self.multi:
            input_size =  input_size // len(action_space)
            num_outputs = num_outputs // len(action_space)
        hidden_size: int = 100
        self._network: nn.Sequential = nn.Sequential(
            nn.Flatten(1,-1),
            nn.Linear(in_features=input_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=num_outputs),
        )

class CNN(_BaseModel):
    """
        Convolutional Neural Network Model
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs,model_config, name)
        self._network: nn.Sequential = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=100,kernel_size=(1,72,40),stride=(1, 72, 40)),
            nn.ReLU(),
            nn.Conv3d(in_channels=100,out_channels=20,kernel_size=(1,1,1)),
            nn.ReLU(),
            nn.Flatten(3,-1),
            nn.Conv2d(in_channels=20,out_channels=5,kernel_size=(1,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=5*obs_space.shape[1],out_features=num_outputs),
        )        
    

# class TorchRNNModel(RecurrentNetwork, nn.Module):
#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#         fc_size=64,
#         lstm_state_size=256,
#     ):
#         nn.Module.__init__(self)
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)

#         # self.obs_size = get_preprocessor(obs_space)(obs_space).size
#         self.fc_size = fc_size
#         self.lstm_state_size = lstm_state_size

#         # Build the Module from fc + LSTM + 2xfc (action + value outs).
#         self.fc1 = nn.Linear(self.obs_size, self.fc_size)
#         self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
#         self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
#         self.value_branch = nn.Linear(self.lstm_state_size, 1)
#         # Holds the current "base" output (before logits layer).
#         self._features = None

#     @override(ModelV2)
#     def get_initial_state(self):
#         # TODO: (sven): Get rid of `get_initial_state` once Trajectory
#         #  View API is supported across all of RLlib.
#         # Place hidden states on same device as model.
#         h = [
#             self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#             self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#         ]
#         return h

#     @override(ModelV2)
#     def value_function(self):
#         assert self._features is not None, "must call forward() first"
#         return torch.reshape(self.value_branch(self._features), [-1])

#     @override(RecurrentNetwork)
#     def forward_rnn(self, inputs, state, seq_lens):
#         """Feeds `inputs` (B x T x ..) through the Gru Unit.
#         Returns the resulting outputs as a sequence (B x T x ...).
#         Values are stored in self._cur_value in simple (B) shape (where B
#         contains both the B and T dims!).
#         Returns:
#             NN Outputs (B x T x ...) as sequence.
#             The state batches as a List of two items (c- and h-states).
#         """
#         x = nn.functional.relu(self.fc1(inputs))
#         self._features, [h, c] = self.lstm(
#             x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
#         )
#         action_out = self.action_branch(self._features)
#         return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]