import numpy as np
from mdp.abstract_mdp import AbstractMDP
from model_free_prediction.abstract_evaluator import AbstractEvaluator
from policy import AbstractPolicy


class TDLambdaEvaluator(AbstractEvaluator):
    def __init__(self,
                 env: AbstractMDP,
                 alpha: float,
                 lambd: float):
        """
        Initializes the TD(λ) Evaluator.
        :param env: a mdp object
        :param alpha: the step size
        :param lambd: the trace decay parameter (λ)
        """
        self.env = env
        self.alpha = alpha
        self.lambd = lambd
        #estimate of state-value function
        self.value_fun = np.zeros(self.env.num_states)
        self.eligibility_traces = np.zeros(self.env.num_states)

    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        """
        Perform the TD prediction algorithm.
        :param policy: a policy object that provides action probabilities for each state
        :param num_episodes: number of episodes to run for estimating V(s)
        :return: the state-value function V(s) for the associated policy
        """
        #reset value function
        self.value_fun.fill(0)

        for _ in range(num_episodes):
            self._update_value_function(policy)

        return self.value_fun.copy()

    def _update_value_function(self, policy: AbstractPolicy) -> None:
        """
        Runs a single episode using the TD(λ) method to update the value function
        :param policy: a policy object that provides action probabilities for each state
        """
        state=self.env.reset()
        self.eligibility_traces.fill(0)
        done=False
        while not done:
            action=policy.sample_action(state)
            next_state,reward,done=self.env.step(action)
            if done:
                td_error = reward-self.value_fun[state]
            else:
                td_error=reward+self.env.discount_factor*self.value_fun[next_state]-self.value_fun[state]
            self.eligibility_traces[state] += 1
            self.value_fun += self.alpha*td_error*self.eligibility_traces
            self.eligibility_traces*=self.env.discount_factor*self.lambd
            state=next_state
