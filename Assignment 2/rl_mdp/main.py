import numpy as np

from rl_mdp.bellman.bellmanequationsolver import BellmanEquationSolver
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.mdp.mdp import MDP
from rl_mdp.mdp.reward_function import RewardFunction
from rl_mdp.mdp.transition_function import TransitionFunction
from rl_mdp.policy.policy import Policy


def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    # Part 1
    mdp = create_mdp()
    policy = Policy()
    for state in mdp.states:
        policy.set_action_probabilities(state, [0.4, 0.6])
    bes = BellmanEquationSolver(mdp)
    value_function = bes.policy_evaluation(policy)
    print(value_function)

def create_mdp() -> AbstractMDP:
    rewards = {
        (0, 0): -0.9,
        (0, 1): -0.9,
        (1, 0): -0.1,
        (1, 1): -0.1,
        (2, 0): -1,
        (2, 1): -0.1
    }

    reward_function = RewardFunction(rewards)

    transitions = {
        (0, 0): np.array([0.1, 0.0, 0.9]),
        (0, 1): np.array([0.1, 0.9, 0.0]),

        (1, 0): np.array([0.9, 0.1, 0.0]),
        (1, 1): np.array([0.9, 0.1, 0.0]),

        (2, 0): np.array([0.0, 0.9, 0.1]),
        (2, 1): np.array([0.9, 0.0, 0.1])
    }

    transition_function = TransitionFunction(transitions)

    return MDP(states=[0,1,2], actions=[0, 1],
               transition_function=transition_function,
               reward_function=reward_function,
               discount_factor=0.9)

if __name__ == "__main__":
    main()