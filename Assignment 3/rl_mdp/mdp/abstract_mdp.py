from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class AbstractMDP(ABC):
    """
    Simple MDP abstract base class.
    Assumes: discrete state and action space (represented as integers).
    """
    @abstractmethod
    def reset(self) -> int:
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool]:
        pass

    @abstractmethod
    def transition_prob(self, new_state: int, state: int, action: int) -> float |  np.ndarray:
        """
        An MDP should have a transition function. In this case modeled as p(s'|s,a).
        :param new_state: state s'
        :param state: state s
        :param action: action a
        :return: Probability p(s'|s,a)
        """
        pass

    @abstractmethod
    def rewards(self, state: int, action: int) -> float:
        """
        An MDP should have a reward function. In this case modeled as r(s,a).
        :param state: state s
        :param action: action a
        :return: reward r(s,a)
        """
        pass

    @property
    @abstractmethod
    def states(self) -> List[int]:
        """
        Getter for actions.
        :return: the list of all possible states represented by a list of actions
        """
        pass

    @property
    @abstractmethod
    def discount_factor(self) -> float:
        """
        Getter for discount factor.
        :return: the discount factor
        """
        pass

    @property
    @abstractmethod
    def num_states(self) -> int:
        """
        Getter for number of states.
        :return: the number of states
        """
        pass


    @property
    @abstractmethod
    def num_actions(self) -> int:
        """
        Getter for number of actions.
        :return: the number of actions
        """
        pass