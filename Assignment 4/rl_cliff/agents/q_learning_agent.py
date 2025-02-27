import numpy as np
import gymnasium as gym
from rl_cliff.agents.tabularagent import TabularAgent


class QLearningAgent(TabularAgent):
    def __init__(self,
                 state_space: gym.spaces.Discrete,
                 action_space: gym.spaces.Discrete,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        super().__init__(state_space, action_space, learning_rate, discount_factor)
        self.epsilon = epsilon
        self.agent_type = 'Q-LEARNING'

    def update(self, transition: tuple) -> None:
        obs, action, reward, next_obs = transition
        next_action = np.argmax(self.q_table[next_obs])
        target = reward + self.discount_factor * self.q_table[next_obs, next_action]
        td_error = target - self.q_table[obs, action]
        self.q_table[obs, action] += self.learning_rate * td_error


    def behavior_policy(self, state) -> int:
        if np.random.rand() < self.epsilon:
            action = self.env_action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

    def target_policy(self, state) -> int:
        action = np.argmax(self.q_table[state])
        return action