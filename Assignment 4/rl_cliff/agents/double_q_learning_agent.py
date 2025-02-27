import numpy as np
import gymnasium as gym
from rl_cliff.agents.tabularagent import TabularAgent


class DoubleQLearningAgent(TabularAgent):
    def __init__(self,
                 state_space: gym.spaces.Discrete,
                 action_space: gym.spaces.Discrete,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        super().__init__(state_space, action_space, learning_rate, discount_factor)
        self.q_table_1 = np.zeros([state_space.n, action_space.n])
        self.q_table_2 = np.zeros([state_space.n, action_space.n])
        self.epsilon = epsilon
        self.agent_type = 'DOUBLE-Q-LEARNING'

    def update(self, transition: tuple) -> None:
        obs, action, reward, next_obs = transition
        if np.random.rand() > 0.5:
            next_action = np.argmax(self.q_table_2[next_obs])
            target = reward + self.discount_factor * self.q_table_1[next_obs, next_action]
            td_error = target - self.q_table_2[obs, action]
            self.q_table_2[obs, action] += self.learning_rate * td_error
        else:
            next_action = np.argmax(self.q_table_1[next_obs])
            target = reward + self.discount_factor * self.q_table_2[next_obs, next_action]
            td_error = target - self.q_table_1[obs, action]
            self.q_table_1[obs, action] += self.learning_rate * td_error

    def behavior_policy(self, state) -> int:
        if np.random.rand() < self.epsilon:
            action = self.env_action_space.sample()
        else:
            q_table = self.q_table_1 + self.q_table_2
            action = np.argmax(q_table[state])
        return action

    def target_policy(self, state) -> int:
        q_table = self.q_table_1 + self.q_table_2
        action = np.argmax(q_table[state])
        return action