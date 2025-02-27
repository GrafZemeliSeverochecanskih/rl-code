from typing import Union, SupportsFloat
import gymnasium as gym
import torch
import numpy as np
from rl_policy_gradient.agents.abstractagent import AbstractAgent
from rl_policy_gradient.models.mlp import MLP
from rl_policy_gradient.models.stochastic_policy_network import StochasticPolicyNetwork
from rl_policy_gradient.trainers.actrainer import ACTrainer
from rl_policy_gradient.trainers.reinforce_trainer import ReinforceTrainer
from rl_policy_gradient.util.device import fetch_device


class PolicyGradientAgent(AbstractAgent):
    def __init__(self,
                 state_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 agent_type: str,
                 discount_factor: float = 0.9999999,
                 learning_rate_actor: float = 0.0006,
                 learning_rate_critic: float = 0.006):
        super().__init__(state_space, action_space)
        """
        Initialize a PolicyGradient Agent. This class has a minimal interface.
        The actual algorithm is implemented in the associated Trainer class.
        NOTE: One rule of thumb for the learning rates is that the learning rate of the actor should be lower
        than the critic. Intuitively because the estimated values of the critic are based on past policies,
        so the actor cannot "get ahead" of the critic.
        :param state_space: the state space of the environment
        :param action_space: the action space of the environment
        :param discount_factor: discount factor for future rewards
        :param learning_rate_actor: learning rate for the actor model
        :param learning_rate_critic: learning rate for the critic model
        """
        self._policy = StochasticPolicyNetwork(input_size=state_space.shape[0],
                                               output_size=action_space.shape[0]).to(device=fetch_device())
        #output_size=1 because value function returns a scalar value.
        self._value_fun = MLP(input_size=state_space.shape[0], output_size=1).to(device=fetch_device())

        self.agent_type = agent_type
        self._trainer = None
        if self.agent_type == "ACTOR-CRITIC-AGENT":
            self._trainer = ACTrainer(self._policy,
                                      self._value_fun,
                                      learning_rate_actor,
                                      learning_rate_critic,
                                      discount_factor)
        elif self.agent_type == "REINFORCE-AGENT":
            self._trainer = ReinforceTrainer(self._policy,
                                             self._value_fun,
                                             learning_rate_actor,
                                             learning_rate_critic,
                                             discount_factor)

        self.device = fetch_device()

    def update(self,
               state: Union[torch.Tensor, np.ndarray],
               action: Union[torch.Tensor, np.ndarray],
               reward: Union[torch.Tensor, np.ndarray, SupportsFloat],
               next_state: Union[torch.Tensor, np.ndarray],
               done: bool) -> None:
        """
        Perform a gradient descent step on both actor (policy) and critic (value function).
        """
        self._trainer.add_transition(state, action, reward, next_state, done)
        self._trainer.train()

    def policy(self, state: np.ndarray) -> np.array:
        """
        Get the action to take based on the current state.
        :param state: the current state of the environment
        :return: the action to take
        """
        state = torch.from_numpy(state).to(device=fetch_device(), dtype=torch.float64)
        action, _ = self._policy.sample_action(state)

        #put the tensor back on the CPU (if applicable) and convert to numpy array
        return action.cpu().numpy()

    def save(self, file_path='./') -> None:
        """
        Save the actor and critic models.
        :param file_path: The directory path to save the models
        """
        torch.save(self._policy.state_dict(), file_path + "actor_model.pth")
        torch.save(self._value_fun.state_dict(), file_path + "critic_model.pth")

    def load(self, file_path='./') -> None:
        """
        Load the actor and critic models.
        :param file_path: the directory path to load the models from
        """
        self._policy.load_state_dict(torch.load(file_path + "actor_model_best_inv_pendulum.pth"))
        self._value_fun.load_state_dict(torch.load(file_path + 'critic_model_best_inv_pendulum.pth'))