from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
import gym

torch, nn = try_import_torch()
__all__ = ['Preprocessor']

class Preprocessor():
    """
        Define Preprocessor API,
        Wrap Around Function And Enable Shared API.
    """

    def __init__(self, obs_space, options):
        """Constructor

        Args:
            obs_space (gym.spaces): Define The Space Bounds.
            options (dict): The Function & Argeuments.
        """
        assert options != None and 'func' in options
        assert options != None and 'argument_list' in options
        
        self.func = options['func']
        self.argument_list = options['argument_list']
            
    def transform(self,observation):
        """Transfrom Observation To New One.

        Args:
            observation (TensorType): The Observation.

        Returns:
            TensorType: The New Observation After Transfromed.
        """
        return self.func(observation,*self.argument_list)
