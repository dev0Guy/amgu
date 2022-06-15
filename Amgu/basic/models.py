import collections
import numpy as np

from abc import abstractmethod


class _Base(object):
    def __init__(self, action_spce):
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
        super().__init__(action_space)
        self.action_space = action_space

    def predict(self, observation, deterministic=False):
        """_summary_

        Args:
            observation (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.action_space.sample()
