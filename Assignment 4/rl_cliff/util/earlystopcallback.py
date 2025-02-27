from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStopOnNegativeRewardCallback(BaseCallback):
    """
    A custom callback that stops training early if the reward is -100.
    """

    def __init__(self, stop_reward=-100, verbose=0):
        super(EarlyStopOnNegativeRewardCallback, self).__init__(verbose)
        self.stop_reward = stop_reward

    def _on_step(self) -> bool:
        """
        This method will be called by the model after every step.
        We check if the reward is equal to `self.stop_reward` and stop training if true.
        """
        rewards = self.locals['rewards']

        #if any reward equals the stop_reward, manually terminate the episode and reset
        if any(reward == self.stop_reward for reward in rewards):
            if self.verbose > 0:
                logger.info(f"Resetting environment due to reward condition met: {self.stop_reward}.")

            #mark the episode as done to trigger reset
            #forcefully end the episode
            self.locals['dones'] = [True] * len(self.locals['dones'])

            return True

        return True