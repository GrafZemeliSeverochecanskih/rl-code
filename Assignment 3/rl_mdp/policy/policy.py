from typing import List, Optional
import numpy as np
from policy import AbstractPolicy


class Policy(AbstractPolicy):
    """
    This class acts as a wrapper that maps states to a probability vector for each action.
    In case of a deterministic policy we would have that one action gets probability one.
    """
    def __init__(self, policy_mapping: Optional[np.ndarray] = None, num_actions: Optional[int] = None):
        """
        Initializes a simple (stochastic) policy.
        :param policy_mapping: a NumPy array where each element represents a deterministic action for each state.
                               This will be converted to a stochastic policy where one action has probability 1
        :param num_actions: number of possible actions (required if policy_mapping is provided)
        """
        #dictionary to store state-action probability distributions
        self.action_dist = {}

        if policy_mapping is not None:
            if num_actions is None:
                raise ValueError("You need to pass a num_actions argument when passing a policy_mapping.")
            for state, action in enumerate(policy_mapping):
                self.set_action_probabilities(state, [1.0 if a == action else 0.0 for a in range(num_actions)])

    def sample_action(self, state: int) -> int:
        """
        Samples an action from the policy given the current state.
        :param state: the state for which an action should be sampled
        :return: the sampled action
        """
        action_probabilities = self._action_probabilities(state)
        #implicit assumption that actions are represented as 0, 1, 2, ..., |A|
        return np.random.choice(len(action_probabilities), p=action_probabilities)

    def set_action_probabilities(self, state: int, action_probabilities: List[float]) -> None:
        """
        Sets the action probabilities for a given state in the policy.
        :param state: the state for which the action probabilities should be set
        :param action_probabilities: a list representing the probability distribution over actions
        """
        if abs(sum(action_probabilities) - 1.0) > 1e-6:
            raise ValueError("The action probabilities must sum to 1.")
        self.action_dist[state] = action_probabilities

    def action_prob(self, state: int, action: int) -> float:
        """
        :param state: state s
        :param action: action a
        :return: gets the probability of an action `a` given the state `s` p(a|s)
        """
        return self._action_probabilities(state)[action]

    def _action_probabilities(self, state: int) -> List[float]:
        """
        Gets the action probabilities for a given state as per the current policy.
        :param state: the state for which the action probabilities are requested
        :return: a list representing the probability distribution over actions for the given state
        :raises ValueError: If the state does not have action probabilities set
        """
        if state in self.action_dist:
            return self.action_dist[state]
        else:
            raise ValueError(f"No action probabilities defined for state {state}.")

    def __str__(self) -> str:
        """
        Returns a string representation of the policy object, showing the action probability distributions
        for each state.
        :return: a string representation of the policy
        """
        policy_str = "Policy:\n"
        for state, action_probs in self.action_dist.items():
            policy_str += f"  State {state}: {action_probs}\n"
        return policy_str if self.action_dist else "Policy: No action probabilities defined."