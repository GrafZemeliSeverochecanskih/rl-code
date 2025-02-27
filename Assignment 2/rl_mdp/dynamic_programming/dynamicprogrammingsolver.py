import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.policy.abstract_policy import AbstractPolicy
from rl_mdp.policy.policy import Policy



class DynamicProgrammingSolver:
    def __init__(self, mdp: AbstractMDP, theta: float = 1e-6):
        """
        Initializes the Dynamic Programming Solver for the given MDP.
        :param mdp: An instance of a class that implements AbstractMDP.
        :param theta: Convergence threshold for iterative methods.
        """
        self.mdp = mdp
        self.theta = theta

    def value_iteration(self) -> Policy:
        """
        Performs value iteration to find the optimal policy.
        :return: An optimal policy.
        """
        n_states = len(self.mdp.states)
        value_vec = np.zeros(n_states)
        while True:
            delta = np.float64(0.0)
            # Compute new values for al states
            for state in self.mdp.states:
                prev_state_value = value_vec[state]
                state_value = np.max([
                    self.mdp.reward(state,action) + self.mdp.discount_factor *
                    np.sum([self.mdp.transition_prob(next_state, state, action)
                            * value_vec[next_state]
                            for next_state in self.mdp.states])
                    for action in self.mdp.actions
                ])
                delta = max([delta, np.abs(state_value - prev_state_value)])
                value_vec[state] = state_value

            if delta < self.theta:
                break

        optimal_policy = self.policy_improvement(value_vec)

        return Policy(policy_mapping=optimal_policy, num_actions=len(self.mdp.actions))

    def policy_iteration(self) -> AbstractPolicy:
        """
        Performs policy iteration to find the optimal policy.
        :return: An optimal policy.
        """
        n_states = len(self.mdp.states)
        policy = np.random.choice(self.mdp.actions, size=n_states)
        while True:
            value_vec = self.iterative_policy_evaluation(
                    Policy(policy, len(self.mdp.actions)))
            new_policy = self.policy_improvement(value_vec)
            if np.array_equal(new_policy, policy):
                break
            policy = new_policy
        return Policy(policy_mapping=policy, num_actions=len(self.mdp.actions))


    def iterative_policy_evaluation(self, policy: AbstractPolicy) -> np.ndarray:
        """
        Evaluates iteratively the value function for a given policy.

        :param policy: An instance of the Policy class, which provides the action probabilities for each state.
        :return: A NumPy array representing the value function for the given policy.
        """
        n_states = len(self.mdp.states)
        value_vec = np.zeros(n_states)
        while True:
            delta = np.float64(0.0)
            for state in self.mdp.states:
                prev_state_value = value_vec[state]
                state_value = np.sum([
                    policy.action_prob(state, action) *
                    (self.mdp.reward(state, action) + self.mdp.discount_factor *
                     np.sum([self.mdp.transition_prob(next_state, state, action)
                             * value_vec[next_state]
                             for next_state in self.mdp.states]))
                    for action in self.mdp.actions
                ])
                delta = np.max([delta, np.abs(state_value - prev_state_value)])
                value_vec[state] = state_value

            if delta < self.theta:
                break
        return value_vec


    def policy_improvement(self, value_vec: np.ndarray) -> AbstractPolicy:
        """
        Performs policy improvement on a given policy.

        :return: A policy.
        """
        n_states = len(self.mdp.states)
        new_policy = np.zeros(n_states, dtype=int)
        for state in self.mdp.states:
            action_values = np.array([
                self.mdp.reward(state, action) + self.mdp.discount_factor *
                np.sum([self.mdp.transition_prob(next_state, state, action)
                        * value_vec[next_state]
                        for next_state in self.mdp.states])
                    for action in self.mdp.actions
            ])
            new_policy[state] = np.argmax(action_values)
        return new_policy