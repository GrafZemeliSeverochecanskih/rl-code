from stable_baselines3.common.base_class import BaseAlgorithm
from rl_cliff.agents.abstractagent import AbstractAgent


class StableBaselinesAdapter(AbstractAgent):
    """
    Adapter class to use Stable Baselines models as agents.
    """

    def __init__(self, model: BaseAlgorithm) -> None:
        """
        Constructor for StableBaselinesAdapter.
        :param model: stable Baselines model to adapt
        """
        super().__init__()
        self._sb_model = model

    def update(self, transition: tuple) -> None:
        # Stable Baselines handles its own training, so nothing to implement here.
        pass

    def target_policy(self, state) -> int:
        """
        Returns the action from the Stable Baselines model's learned policy,
        representing the target policy (typically deterministic during evaluation).
        :param state: the current state of the environment
        :return: action chosen by the policy (deterministic action)
        """
        action, _states = self._sb_model.predict(state, deterministic=True)
        return int(action)

    def behavior_policy(self, state) -> int:
        """
        Returns the action from the Stable Baselines model's behavior policy.
        The behavior policy may use exploration (stochastic action selection).
        :param state: the current state of the environment
        :return: action chosen by the policy (stochastic action)
        """
        action, _states = self._sb_model.predict(state, deterministic=False)
        return int(action)