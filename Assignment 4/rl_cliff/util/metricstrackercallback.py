import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from rl_cliff.agents.sb_adapter import StableBaselinesAdapter
from rl_cliff.util.evaluation import evaluate_target_policy
from rl_cliff.util.metricstracker import MetricsTracker


class MetricsTrackerCallback(BaseCallback):
    """
    Stable Baselines callback to track and record the agent's performance using the MetricsTracker.
    Additionally, it runs the evaluate_target_policy function at the end of each episode.
    """

    def __init__(self, behavioral_tracker: MetricsTracker,
                 target_tracker: MetricsTracker,
                 eval_env: gym.Env,
                 agent_id: str = "DQN", verbose: int = 0):
        """
        Constructor for StableBaselinesMetricsCallback.
        :param behavioral_tracker: the MetricsTracker instance for tracking behavioral policy metrics
        :param target_tracker: the MetricsTracker instance for tracking target policy metrics
        :param eval_env: the evaluation environment
        :param agent_id: the identifier for the agent (default: "DQN")
        :param verbose: verbosity level
        """
        super().__init__(verbose)
        self.agent_id = agent_id
        self.behavioral_tracker = behavioral_tracker
        self.target_tracker = target_tracker
        self.eval_env = eval_env
        self.episode_reward = 0
        self.episode_idx = 0  # Track the current episode index
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

        :return: If the callback returns False, training is aborted early.
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
        #record the return for the behavioral policy
        self.behavioral_tracker.record_return(self.agent_id, self.episode_idx, self.episode_reward)

        #get the current mean and stddev return for this episode
        current_mean_return, current_std_return = self.behavioral_tracker.get_mean_std_return(self.agent_id,
                                                                                              self.episode_idx)
        if current_mean_return and current_mean_return > self.highest_avg_return:
            self.highest_avg_return = current_mean_return

        agent = StableBaselinesAdapter(self.model)
        agent.agent_type = self.agent_id

        #evaluate the target policy using the provided evaluation environment
        evaluate_target_policy(self.eval_env, agent, self.target_tracker, self.episode_idx)

        #reset the episode reward
        self.episode_reward = 0
        self.episode_idx += 1  # Increment the episode index

    def _on_training_end(self) -> None:
        #you could add additional logging or save the tracker data to file if needed
        pass