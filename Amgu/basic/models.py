import collections
import numpy as np
import torch as th
from abc import abstractmethod


class _Base(object):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, observation, deterministic=False):
        pass


class Random(_Base):
    """_summary_

    Args:
        _Base (_type_): _description_
    """

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def predict(self, observation, deterministic=False):
        """_summary_

        Args:
            observation (_type_): _description_

        Returns:
            _type_: _description_
        """
        return observation, self.action_space.sample()


class Attack(_Base):
    """_summary_

    Args:
        _Base (_type_): _description_
    """

    def __init__(self, attack, org_model):
        super().__init__()
        self.attack = attack    
        self.org_model = org_model

    def _attack(self, observation, deterministic):
        model = self.org_model.q_net.q_net
        obs_attack = th.from_numpy(obs_attack)
        action_shape =  self.org_model.predict(observation)
        target = np.zeros(action_shape)
        obs_attack = self.attack(model, obs_attack[None,:], th.from_numpy(target))[0,:]
        return obs_attack
    
    def predict(self, observation, deterministic=False):
        """_summary_
        Args:
            observation (_type_): _description_
        Returns:
            _type_: _description_
        """
        obs = self._attack(observation,deterministic)
        attack_action, _ = self.org_model.predict(obs,deterministic)
        return obs,attack_action