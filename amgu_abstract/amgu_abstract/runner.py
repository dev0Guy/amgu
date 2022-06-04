from abc import abstractmethod
from .model import *

__all__ = ["RunnerWrapper"]


class RunnerWrapper:
    """
    Define Runner API,
    Enable Shared API.
    """

    def __init__(self, config, model, env, agent):
        """Constructor

        Args:
            config (dict): Runner Config, All Information Needed For Build.
            model (Torch|Algorithem Wrapper): The 'Brain'.
            env (gym.Env): The Environment To Run On.
            agent (*): Instnace Of Agent To Run On.
        """
        class_options = TorchWrapper
        # assert issubclass(model, class_options)
        self.agent = agent
        self.config = config
        self.model = model
        self.env = env

    @abstractmethod
    def train(self):
        """Train The Agent & Model. Need To Be Implemented In Child."""
        pass

    @abstractmethod
    def eval(self):
        """Evalouate The Agent & Model. Need To Be Implemented In Child."""
        pass
