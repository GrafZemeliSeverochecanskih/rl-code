from typing import  Optional
import numpy as np
from mdp import RewardFunction
from mdp import TransitionFunction

class MDP(AbstractMDP):
    def __init__(
            self,
            states: List[int],
            actions: List[int],
            transition_function: TransitionFunction,
            reward_function: RewardFunction,
            discount_factor: float = 0.9,
            terminal_state: Optional[int] = None,
            start_state: Optional[int] = 0
    ):
        """
        Initializes the Markov Decision Process (MDP).
        :param states: a list of states in the MDP
        :param actions: a list of actions in the MDP
        :param transition_function: a TransitionFunction object that provides transition probabilities
        :param reward_function: a RewardFunction object that provides rewards
        :param discount_factor: a discount factor for future rewards
        :param terminal_state: a terminal state
        :param start_state: a starting state. If set, then reset() will always return that state
        """
        self._states = states
        self._actions = actions
        self._transition_function = transition_function
        self._reward_function = reward_function
        self._discount_factor = discount_factor

        self._start_state = start_state
        self._curr_state = self._start_state if self._start_state is not None else np.random.choice(self._states)
        #assuming one terminal state for simplicity
        self._terminal_state = terminal_state

    def reset(self) -> int:
        """
        Re-initialize the state by sampling uniformly from the state space.
        :return: new initial state
        """
        self._curr_state = self._start_state if self._start_state is not None else np.random.choice(self._states)
        return self._curr_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Perform a realization of p(s'|s,a) and r(s,a).
        :param action: action taken by the agent
        :return:a tuple containing the new state, the reward, and a done flag
        """
        #get the transition probabilities for the current state and action
        transition_probs = self._transition_function(self._curr_state, action)

        #sample the next state based on the transition probabilities
        next_state = np.random.choice(self._states, p=transition_probs)

        #calculate the reward for the current state and action
        reward = self._reward_function(self._curr_state, action)

        self._curr_state = next_state
        done = False if self._terminal_state is None else next_state == self._terminal_state
        return next_state, reward, done

    def transition_prob(self, new_state: int, state: int, action: int) -> float | np.ndarray:
        """
        Returns the transition probabilities for the new state given state and action by
        calling the transition function.
        :param new_state: new state
        :param state: current state
        :param action: action taken
        :return: probability p(s'|s,a)
        """
        return self._transition_function(state, action)[new_state]

    def rewards(self, state: int, action: int) -> float:
        """
        Returns the reward for a given state and action by calling the reward function.
        :param state: current state
        :param action: action taken
        :return: a float representing the reward for the given (state, action) pair
        """
        return self._reward_function(state, action)

    @property
    def states(self) -> List[int]:
        """
        Getter for the list of states.
        :return: a list of states
        """
        return self._states

    @property
    def actions(self) -> List[int]:
        """
        Getter for the list of actions.
        :return: a list of actions
        """
        return self._actions

    @property
    def discount_factor(self) -> float:
        """
        Getter for the discount factor.
        :return: the discount factor
        """
        return self._discount_factor

    @property
    def num_states(self) -> int:
        """
        Getter for the number of states.
        :return: the number of states
        """
        return len(self._states)

    @property
    def num_actions(self) -> int:
        """
        Getter for the number of actions.
        :return: the number of actions
        """
        return len(self._actions)

    @property
    def current_state(self) -> int:
        """
        Getter for the number of actions.
        :return: the current state
        """
        return self._curr_state
