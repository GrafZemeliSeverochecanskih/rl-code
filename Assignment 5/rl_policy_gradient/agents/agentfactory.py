import gymnasium as gym
from typing import Optional
from rl_policy_gradient.agents.abstractagent import AbstractAgent
from rl_policy_gradient.agents.policy_gradient_agent import PolicyGradientAgent
from rl_policy_gradient.agents.randomagent import RandomAgent


class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """

    @staticmethod
    def create_agent(agent_type: str, env: gym.Env, *,
                     lr_actor: Optional[float] = None,
                     lr_critic: Optional[float] = None) -> AbstractAgent:
        """
        Factory method for Agent creation.
        :param env: gymnasium environment
        :param agent_type: a string key corresponding to the agent
        :param lr_actor: learning rate of policy approximation
        :param lr_critic: learning rate of value function approximation
        You could pass a dictionary if you want to generalize this method.
        :return: an object of type Agent
        """
        obs_space = env.observation_space
        action_space = env.action_space

        if agent_type == "ACTOR-CRITIC-AGENT" or agent_type == "REINFORCE-AGENT":
            return PolicyGradientAgent(
                obs_space,
                action_space,
                agent_type,
                learning_rate_actor=lr_actor,
                learning_rate_critic=lr_critic
            )
        elif agent_type == "RANDOM":
            return RandomAgent(obs_space, action_space)

        raise ValueError("Invalid agent type")