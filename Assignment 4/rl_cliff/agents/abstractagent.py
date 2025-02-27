from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """
    Agent abstract base class.
    """
    def __init__(self):
        self.agent_type = "NONE"

    @abstractmethod
    def update(self, transition: tuple) -> None:
        """
        Where the update rule is applied
        :param transition: (S, A, R, S') (Q-learning), (S, A, R, S', A') (SARSA).
        """
        pass

    #for the purposes of the assignment we make an explicit distinction between
    #the target and behavioral policy.
    #an alternative implementation would be to have a policy method with an
    #additional boolean parameter like 'determinstic=True'.

    @abstractmethod
    def behavior_policy(self, state) -> int:
        """
        This is where you would do action selection for the behavior.
        :param state: given state
        :return an action
        """
        pass

    @abstractmethod
    def target_policy(self, state) -> int:
        """
        This is where you would do action selection for the target policy.
        :param state: given state
        :return an action
        """
        pass
