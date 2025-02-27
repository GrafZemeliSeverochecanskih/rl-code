import gymnasium as gym
from rl_cliff.agents.double_q_learning_agent import DoubleQLearningAgent
from rl_cliff.agents.q_learning_agent import QLearningAgent
from rl_cliff.agents.random_agent import RandomAgent
from rl_cliff.agents.sarsa_agent import SarsaAgent
from rl_cliff.agents.tabularagent import TabularAgent


class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """

    @staticmethod
    def create_agent(agent_type: str, env: gym.Env, *, eps: float, lr: float) -> TabularAgent:
        """
        Factory method for Agent creation.
        :param env: gymnasium environment
        :param agent_type: a string key corresponding to the agent
        :param eps: epsilon parameter
        :param lr: learning rate parameter
        :return: an object of type Agent
        """
        obs_space = env.observation_space
        action_space = env.action_space

        if agent_type == "SARSA":
            return SarsaAgent(obs_space, action_space, epsilon=eps, learning_rate=lr)
        elif agent_type == "Q-LEARNING":
            return QLearningAgent(obs_space, action_space, epsilon=eps, learning_rate=lr)
        elif agent_type == "DOUBLE-Q-LEARNING":
            return DoubleQLearningAgent(obs_space, action_space, epsilon=eps, learning_rate=lr)
        elif agent_type == "RANDOM":
            return RandomAgent(obs_space, action_space)

        raise ValueError("Invalid agent type")