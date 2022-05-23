from abc import abstractmethod

__all__ = ["RewardWrapper"]


class RewardWrapper:
    """
    Define RewarD API,
    Wrap Around Function And Enable Shared API.
    """

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get(self, observation):
        """Return The Reward, Need To Be Implemented In Child.
        Args:
            observation (numpy.ndarray): the observation itself
        """
        pass
