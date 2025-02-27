from abc import ABC, abstractmethod
from typing import Union, SupportsFloat
import gymnasium as gym
import torch
import numpy as np


class AbstractAgent(ABC):
    """
    Agent abstract base class.
    """

    def __init__(self,
                 state_space: gym.spaces.Box,
                 action_space: gym.spaces.Box):
        """
        Agent Base Class constructor.
        :param state_space: state space of the gym environment
        :param action_space: action space of the gym environment
        """
        self.state_space = state_space
        self.action_space = action_space

    @abstractmethod
    def update(self,
               state: Union[torch.Tensor, np.ndarray],
               action: Union[torch.Tensor, np.ndarray],
               reward: Union[torch.Tensor, np.ndarray, SupportsFloat],
               next_state: Union[torch.Tensor, np.ndarray],
               done: bool) -> None:
        """
        Abstract method where the update rule is applied.
        """
        pass

    @abstractmethod
    def policy(self, state: np.ndarray) -> np.array:
        """
        Abstract method to define the agent's policy.
        For actor critic algorithms the output of the actor would be a probability distribution over actions.
        For discrete actions this is simply a discrete probability distributions, describing a probability
        for each action. For continuous actions you can have some kind of continuous distribution you sample
        actions from.
        :param state: the current state of the environment
        """
        pass

    @abstractmethod
    def save(self, file_path: str = './') -> None:
        """
        Abstract method to save the agent's model.
        :param file_path: The path to save the model
        """
        pass

    @abstractmethod
    def load(self, file_path: str = './') -> None:
        """
        Abstract method to load the agent's model.
        :param file_path: The path to load the model from
        """
        pass