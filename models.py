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
class _ModelWrapper(TorchModelV2, nn.Module):
    """
        Simple Fully Connected Network Model
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)        
        
        obs_type: type = type(obs_space)
        action_type: type = type(action_space)
        
        assert obs_type in [gym.spaces.box.Box,gym.spaces.Dict]
        assert action_type in [gym.spaces.MultiDiscrete,gym.spaces.Discrete,gym.spaces.Dict]
        
        self.multi = action_type is gym.spaces.Dict
        self._output: torch.Tensor = None  
        
    def _preprocess(self,obs: torch.Tensor):
        return obs
            
    @override(ModelV2)
    def forward(self, input_dict: dict, state: list, seq_lens): 
        input = input_dict["obs"]
        input = self._preprocess(input)
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

class FCN(_ModelWrapper):
    """
        Simple Fully Connected Network Model
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs,model_config, name)
        input_size: int = np.prod(obs_space.shape)
        hidden_size: int = 100
        self._network: nn.Sequential = nn.Sequential(
            nn.Flatten(1,-1),
            nn.Linear(in_features=input_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=num_outputs),
        )

class CNN(_ModelWrapper):
    """
        Convolutional Neural Network Model
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs,model_config, name)
        self._network: nn.Sequential = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=100,kernel_size=(1,32,1),stride=(1, 32, 1)),
            nn.MaxPool3d(kernel_size=(1,1,2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=100,out_channels=20,kernel_size=(1,1,20)),
            nn.ReLU(),
            nn.Flatten(3,-1),
            nn.Conv2d(in_channels=20,out_channels=5,kernel_size=(1,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=5*obs_space.shape[1],out_features=500),
            nn.Linear(in_features=5*obs_space.shape[1],out_features=250),
            nn.Linear(in_features=5*obs_space.shape[1],out_features=num_outputs),
        )        
    
class Prototype(_ModelWrapper):
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs,model_config, name)
        num_outputs = int(num_outputs//obs_space.shape[1])
        hidden_size = 10
        self._network = nn.Sequential(
            nn.Linear(in_features=obs_space.shape[2],out_features=hidden_size),
            nn.Linear(in_features=hidden_size,out_features=num_outputs),
            nn.Flatten(1,-1),
        )

    def _preprocess(self,obs: torch.Tensor):
        obs = obs[:,0]
        new_size = obs.size()[:-1]
        new_size = (*new_size,-1)
        obs = obs.reshape(new_size)
        obs =  torch.count_nonzero(obs,dim=-1).float()
        obs = obs.reshape(obs.size()[0],obs.size()[1],-1)
        return obs

class Queue():
    """
        Convolutional Neural Network Model
    """
    def __init__(self,action_impact:list):
        if not action_impact:
            raise Warning('Action Impact Cant be None')
        self.action_impact = action_impact

    def __call__(self,obs: torch.Tensor):
        obs = self._preprocess(obs)
        output = [] 
        for sample in obs:
            action_lst = list()
            for i_idx,intersection in enumerate(sample):
                phase_summer = dict()
                for l_idx,lane_count in enumerate(intersection):
                    if l_idx not in self.action_impact[i_idx]:
                        continue
                    phase = self.action_impact[i_idx][l_idx]
                    if phase not in phase_summer:
                        phase_summer[phase] = 0
                    phase_summer[phase]+= lane_count.detach().float()
                phase_summer =  np.array(list(phase_summer.values()))
                action_lst.append(np.argmax(phase_summer))
            action_lst = np.array(action_lst)
            output.append(torch.from_numpy(action_lst))
        output = torch.stack(output)
        return output
    
    def _preprocess(self,obs: torch.Tensor):
        obs = obs[:,0]
        new_size = obs.size()[:-1]
        new_size = (*new_size,-1)
        obs = obs.reshape(new_size)
        obs =  torch.count_nonzero(obs,dim=-1).float()
        obs = obs.reshape(obs.size()[0],obs.size()[1],-1)
        return obs


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