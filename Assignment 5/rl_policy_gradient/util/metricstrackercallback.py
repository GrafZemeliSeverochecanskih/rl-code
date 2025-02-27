import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger

from rl_policy_gradient.util.metricstracker import MetricsTracker


class MetricsTrackerCallback(BaseCallback):
    """
    Stable Baselines callback to track and record the agent's performance using the MetricsTracker.
    Additionally, it runs the evaluate_target_policy function at the end of each episode.
    """

    def __init__(self, tracker: MetricsTracker,
                 agent_id: str = "PPO", verbose: int = 0):
        """
        Constructor for StableBaselinesMetricsCallback.
        :param tracker: the MetricsTracker instance for tracking policy metrics
        :param eval_env: the evaluation environment
        :param agent_id: the identifier for the agent (default: "DQN")
        :param verbose: verbosity level
        """
        super().__init__(verbose)
        self.agent_id = agent_id
        self.tracker = tracker
        self.episode_reward = 0
        #track the current episode index
        self.episode_idx = 0
        self.highest_avg_return = float('-inf')

    def _on_training_start(self) -> None:
        """
        Initialize metrics tracking when training starts.
        """
        self.episode_reward = 0
        self.highest_avg_return = float('-inf')
        self.episode_idx = 0

    def _on_step(self) -> bool:
        """
        This method will be called after each call to `env.step()`.
        Tracks the reward received during each step.
        :return: If the callback returns False, training is aborted early
        """
        reward = self.locals['rewards']
        dones = self.locals['dones']
        self.episode_reward += reward.item() if isinstance(reward, np.ndarray) else reward

        if np.any(dones):
            self._on_episode_end()

        return True

    def _on_episode_end(self) -> None:
        """
        Record the episode's return and check if the agent achieved a new highest average return.
        """
        logger.info(f"Episode finished, {self.episode_reward}")
        #record the return for the policy
        self.tracker.record_metric("return", agent_id=self.agent_id, episode_idx=self.episode_idx,
                                   value=self.episode_reward)
        #get the current mean and stddev return for this episode
        current_mean_return, current_std_return = self.tracker.get_mean_std("return",
                                                                            self.agent_id,
                                                                            self.episode_idx)
        if current_mean_return and current_mean_return > self.highest_avg_return:
            self.highest_avg_return = current_mean_return

        #reset the episode reward
        self.episode_reward = 0
        #increment the episode index
        self.episode_idx += 1

    def _on_training_end(self) -> None:
        #you could add additional logging or save the tracker data to file if needed
        pass