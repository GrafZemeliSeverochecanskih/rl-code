from abc import ABC
import numpy as np
import gymnasium as gym
from rl_cliff.agents.abstractagent import AbstractAgent


class TabularAgent(AbstractAgent, ABC):

    def __init__(self, state_space: gym.spaces.Discrete, action_space: gym.spaces.Discrete, learning_rate=0.1,
                 discount_factor=0.9):
        """
        Agent Base Class constructor.
        Assumes discrete gymnasium spaces.
        You may want to make these attributes private.
        :param state_space: state space of gymnasium environment
        :param action_space: action space of gymnasium environment
        :param learning_rate: of the underlying algorithm
        :param discount_factor: discount factor (`gamma`)
        """
        super().__init__()
        self.q_table = np.zeros([state_space.n, action_space.n])
        self.env_action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

