from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class RLTrainer(ABC):
    """
    Abstract base class for trainers responsible for performing optimization processes.
    """
    @abstractmethod
    def add_transition(self,
                       state: Union[torch.Tensor, np.ndarray],
                       action: Union[torch.Tensor, np.ndarray],
                       reward: Union[torch.Tensor, np.ndarray],
                       next_state: Union[torch.Tensor, np.ndarray],
                       done: bool) -> None:
        pass

    @abstractmethod
    def train(self) -> bool:
        """
        Perform the optimization process or learning of parameters.
        Return True if optimization was performed else False.
        """
        pass