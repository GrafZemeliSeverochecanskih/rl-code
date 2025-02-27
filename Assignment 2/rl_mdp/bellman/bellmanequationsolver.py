import numpy as np

from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.policy.abstract_policy import AbstractPolicy

class BellmanEquationSolver():
    def __init__(self, mdp: AbstractMDP):
        """
        Initializes the Bellman Equation Solver for the given MDP.
        :param mdp: An instance of a class that implements AbstractMDP.
        """
        self.mdp = mdp

    def policy_evaluation(self, policy: AbstractPolicy) -> np.ndarray:
        """
        Evaluates the value function for a given policy.
        :param policy: An instance of the Policy class, which provides the action probabilities for each state.
        :return: A NumPy array representing the value function for the given policy.
        """
        n_states = len(self.mdp.states)

        #Initialize the reward vector and transition matrix.
        reward_vec = np.zeros(n_states)
        transition_matrix = np.zeros((n_states, n_states))
        for state in self.mdp.states:
            for action in self.mdp.actions:
                action_prob = policy.action_prob(state, action)
                reward_vec[state] += self.mdp.reward(state, action)

                # Compute the transition probabilities for each next state.
                # #Recall: P_{i,j} = p(s_j|s_i) = \sum_a pi(a|s_i)p(s_j|s_i,a).
                for next_state in self.mdp.states:
                    transition_matrix[state, next_state] += (action_prob * self.mdp.transition_prob(next_state, state, action))

        # Solve the linear system (I - gamma * P_pi) * v = r_pi for v.
        identity_matrix = np.eye(n_states)
        A = identity_matrix - self.mdp.discount_factor * transition_matrix
        value_vec = np.linalg.solve(A, reward_vec)

        return value_vec
