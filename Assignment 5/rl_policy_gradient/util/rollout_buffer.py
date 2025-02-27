from collections import deque
import torch
from typing import Tuple, Optional


class RolloutBuffer:
    def __init__(self, batch_size: Optional[int] = None):
        """
        Buffer to store on-policy trajectories for batched processing.
        :param batch_size: number of transitions to retrieve for the batch. If None, retrieve all
        """
        self.trajectories = deque()
        self.batch_size = batch_size

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor,
             done: torch.Tensor) -> None:
        """
        Add a transition to the buffer.
        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: next state observed after the action
        :param done: terminal state or not
        """
        self.trajectories.append((state, action, reward, next_state, done))

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a batch of transitions from the buffer in the correct order.
        If batch_size is not specified, retrieve all transitions.
        :return: batched tensors for states, actions, rewards, next_states, and dones
        """
        if self.batch_size is None:
            #retrieve all transitions if batch_size is None
            batch_size = len(self.trajectories)
        else:
            batch_size = self.batch_size

        assert len(self.trajectories) >= batch_size, "Not enough transitions in buffer"

        batch = [self.trajectories.popleft() for _ in range(batch_size)]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.trajectories.clear()

    def latest(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the latest transition added to the buffer without removing it.
        :return: the latest transition tuple (state, action, reward, next_state, done)
        """
        assert len(self.trajectories) > 0, "No transitions in buffer"
        return self.trajectories[-1]