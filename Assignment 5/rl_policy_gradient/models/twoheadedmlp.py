from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Distribution


class TwoHeadedMLP(ABC, nn.Module):
    """
    Abstract base class for a two-headed MLP model.
    """

    @abstractmethod
    def predict(self, x: torch.Tensor) -> Distribution:
        """
        Compute and return the distribution output of the model given an input tensor.
        :param x: input tensor
        :return: a distribution representing the output of the model
        """
        pass