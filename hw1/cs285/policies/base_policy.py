import abc
import numpy as np


class BasePolicy(object, metaclass=abc.ABCMeta):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs):
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError
