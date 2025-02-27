from typing import Union
import numpy as np
import torch
from torch import nn
from rl_policy_gradient.models.stochastic_policy_network import StochasticPolicyNetwork
from rl_policy_gradient.trainers.rltrainer import RLTrainer
from rl_policy_gradient.util.rollout_buffer import RolloutBuffer


class ACTrainer(RLTrainer):
    """
    One-step Actor-Critic Trainer based on Sutton and Barto's algorithm.
    """

    def __init__(self,
                 policy: StochasticPolicyNetwork,
                 value_fun: nn.Module,
                 learning_rate_actor: float,
                 learning_rate_critic: float,
                 discount_factor: float,
                 batch_size: int = 1):
        """
        Initialize the Actor-Critic Trainer.
        :param policy: the actor model (policy)
        :param value_fun: the critic model (value function)
        :param learning_rate_actor: learning rate for the actor
        :param learning_rate_critic: learning rate for the critic
        :param discount_factor: discount factor for future rewards
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = policy
        self.value_fun = value_fun
        self.discount_factor = discount_factor

        self._batch_size = batch_size
        self.buf = RolloutBuffer(batch_size=self._batch_size)

        # Optimizes policy parameters
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate_actor)
        # Optimizes critic parameters
        self.value_fun_optimizer = torch.optim.Adam(self.value_fun.parameters(), lr=learning_rate_critic)

    def add_transition(self,
                       state: Union[torch.Tensor, np.ndarray],
                       action: Union[torch.Tensor, np.ndarray],
                       reward: Union[torch.Tensor, np.ndarray],
                       next_state: Union[torch.Tensor, np.ndarray],
                       done: bool) -> None:
        """
        Add a transition to the buffer for storing
        :param state: The current state
        :param action: The action taken
        :param reward: The reward received
        :param next_state: The next state after taking the action
        :param done: Whether the episode has ended (boolean)
        """
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float64)
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float64)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.float64)
        next_state_t = torch.as_tensor(next_state, device=self.device, dtype=torch.float64)
        done_t = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        self.buf.push(
            state_t, action_t, reward_t, next_state_t, done_t
        )

    def _compute_loss(self) -> torch.Tensor:
        """
        Compute losses for actor and critic, and write to metrics tracker.
        You can return the sum of the two losses. In terms of gradient flow, as long as
        the losses are independent (i.e., calculated correctly), summing them will properly
        propagate gradients to their respective parameters.
        :return: a tuple of the actor loss and the critic loss
        """
        state, action, reward, next_state, done = self.buf.get()
        current_values = self.value_fun(state).squeeze()
        next_values = self.value_fun(next_state).squeeze() * (1 - done.float())
        td_target = reward + self.discount_factor * next_values
        advantage = (td_target - current_values).detach()
        log_probs = self.policy.log_prob(state, action)
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.MSELoss()(current_values, td_target)
        total_loss = actor_loss + critic_loss

        return total_loss

    def _optimize(self, loss: torch.Tensor) -> None:
        """
        Backpropagate the critic and actor loss.
        :param loss: actor and critic loss
        """
        if loss is None:
            print("Warning: Received None loss in _optimize.")
            return

        #perform backpropagation and update both actor and critic
        self.policy_optimizer.zero_grad()
        self.value_fun_optimizer.zero_grad()
        loss.backward()

        #ensure that gradients flow correctly to their respective parameters
        for param in self.policy.parameters():
            if param.grad is not None:
                #normalize by batch size if necessary
                param.grad /= self._batch_size

        #step both optimizers
        self.policy_optimizer.step()
        self.value_fun_optimizer.step()

    def train(self) -> bool:
        """
        Perform a training step.
        :return: True if optimized occurred, False otherwise
        """
        loss = self._compute_loss()
        self._optimize(loss)
        return True
