from typing import Dict, Tuple

class RewardFunction:
    def __init__(self, rewards: Dict[Tuple[int, int], float]):
        """
        Initializes the reward function with a dictionary.
        :param rewards: a dictionary where keys are (state, action)
                        tuples and values are floats representing the
                        rewards for each (state, action) pair
        """
        self._rewards = rewards

    @property
    def rewards(self):
        return self._rewards

    def __call__(self, state: int, action: int) -> float:
        """
        Returns the rewards for a given state and action
        :param state: current state
        :param action: action taken
        :return: a flot representing the rewards for the given (state, action) pair.
        """
        if (state, action) in self.rewards:
            return self._rewards[(state, action)]
        else:
            raise ValueError(f"No rewards defined for state {state} and action {action}")
