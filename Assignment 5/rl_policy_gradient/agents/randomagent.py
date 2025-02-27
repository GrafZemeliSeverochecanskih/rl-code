from typing import Union, SupportsFloat
import numpy as np
import torch
from rl_policy_gradient.agents.abstractagent import AbstractAgent


class RandomAgent(AbstractAgent):
    def update(self, state: Union[torch.Tensor, np.ndarray], action: Union[torch.Tensor, np.ndarray],
               reward: Union[torch.Tensor, np.ndarray, SupportsFloat], next_state: Union[torch.Tensor, np.ndarray],
               done: bool) -> None:
        pass

    def policy(self, state: np.ndarray) -> np.array:
        return self.action_space.sample()

    def save(self, file_path='./') -> None:
        pass

    def load(self, file_path='./') -> None:
        pass