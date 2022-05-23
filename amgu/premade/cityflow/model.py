import numpy as np
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import gym
from abstract import TorchWrapper,AlgorithmWrapper
# Try Import
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

__all__ = ['CNN','FCN','Qeueue','Random']

class CNN(TorchWrapper):
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs,model_config, name)
        self.network: nn.Sequential = nn.Sequential(
            nn.Conv3d(in_channels=obs_space.shape[0],out_channels=100,kernel_size=(1,32,1),stride=(1, 32, 1)),
            nn.MaxPool3d(kernel_size=(1,1,2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=100,out_channels=20,kernel_size=(1,1,20)),
            nn.ReLU(),
            nn.Flatten(3,-1),
            nn.Conv2d(in_channels=20,out_channels=5,kernel_size=(1,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=5*obs_space.shape[1],out_features=500),
            nn.Linear(in_features=500,out_features=250),
            nn.Linear(in_features=250,out_features=num_outputs),
        ) 


class FCN(TorchWrapper):
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs,model_config, name)
        assert 'intersection_num' in self.model_config
        assert 'hidden_size' in self.model_config
        
        num_outputs = int(num_outputs//self.model_config['intersection_num'])
        hidden_size = self.model_config['hidden_size']
        
        self.network = nn.Sequential(
            nn.Linear(in_features=obs_space.shape[1],out_features=hidden_size),
            nn.Linear(in_features=hidden_size,out_features=num_outputs),
            nn.Flatten(1,-1),
        )


class Qeueue(AlgorithmWrapper):
    def __init__(self,action_impact:list):
        if not action_impact:
            raise Warning('Action Impact Cant be None')
        self.action_impact = action_impact

    def __call__(self,obs: torch.Tensor):
        obs = obs[:,3] 
        obs =  torch.count_nonzero(obs,dim=-1).float()
        output = [] 
        for sample in obs:
            action_lst = list()
            for i_idx,intersection in enumerate(sample):
                phase_summer = np.zeros(max(set(self.action_impact[i_idx].values()))+1)
                for l_idx,lane_count in enumerate(intersection):
                    if l_idx not in self.action_impact[i_idx]:
                        continue
                    phase = self.action_impact[i_idx][l_idx]
                    phase_summer[phase]+= lane_count.detach().float()
                action_lst.append(np.argmax(phase_summer))
            action_lst = np.array(action_lst)
            output.append(torch.from_numpy(action_lst))
        output = torch.stack(output)
        return output


class Random(AlgorithmWrapper):
    def __init__(self,action_impact:list):
        if not action_impact:
            raise Warning('Action Impact Cant be None')
        self.action_impact = action_impact

    def __call__(self,obs: torch.Tensor):
        intersection_num = obs.size()[2]
        return torch.round(torch.rand(intersection_num)*8).int()