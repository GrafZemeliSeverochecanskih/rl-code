from collections import defaultdict
from typing import List, Tuple
import numpy as np
from mdp.abstract_mdp import AbstractMDP
from model_free_prediction.abstract_evaluator import AbstractEvaluator
from policy import AbstractPolicy


class MCEvaluator(AbstractEvaluator):
    def __init__(self, env: AbstractMDP):
        """
        Initializes the Monte Carlo Evaluator.
        :param env: an environment object
        """
        self.env = env
        #estimate of state-value function
        self.value_fun = np.zeros(self.env.num_states)
        #stores returns for each state
        self.returns = defaultdict(list)

    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        """
        Perform the Monte Carlo prediction algorithm.
        :param policy: a policy object that provides action probabilities for each state
        :param num_episodes: number of episodes to run for estimating V(s)
        :return: the state-value function V(s) for the associated policy
        """
        #reset value function
        self.value_fun.fill(0)
        self.returns.clear()

        for _ in range(num_episodes):
            episode = self._generate_episode(policy)
            self._update_value_function(episode)
        return self.value_fun.copy()

    def _generate_episode(self, policy: AbstractPolicy) -> List[Tuple[int, int, float]]:
        """
        Generate an episode following the policy.
        :return: a list of (state, action, reward) tuples representing the episode
        """
        episode = []
        state = self.env.reset()
        done = False

        while not done:
            action = policy.sample_action(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def _update_value_function(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Update the value function using the Monte Carlo method.
        :param episode: a list of (state, action, reward) tuples
        """
        my_return =  0
        idx = len(episode) - 1
        for state, _, reward in reversed(episode):
            my_return = self.env.discount_factor * my_return + reward

            if state not in [ep[0] for ep in episode[:idx]]:
                self.returns[state].append(my_return)
                self.value_fun[state] = np.mean(self.returns[state])
            idx -= 1