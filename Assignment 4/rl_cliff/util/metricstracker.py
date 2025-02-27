import threading
from typing import Union, SupportsFloat, Any, Dict, Tuple
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from rl_cliff.util.welford import Welford


class MetricsTracker:
    """
    Thread-safe object for recording returns across multiple runs.
    Tracks the mean and standard deviation of returns for each episode using Welford's algorithm,
    averaged over multiple runs.
    """

    def __init__(self):
        self._lock = threading.Lock()
        #for each agent, track Welford objects using a dictionary where key is episode_idx
        self._return_history: Dict[str, Dict[int, Welford]] = defaultdict(dict)

    def plot(self, file_name: str, num_episodes: int = None, x_axis_label="Episodes", y_axis_label='Average Return',
             title="Return History") -> None:
        """
        Plot the average return and standard deviation over episodes across multiple runs for each agent.
        """
        with self._lock:
            fig, ax = plt.subplots(figsize=(10, 8))

            for agent_id, episode_welfords in self._return_history.items():
                #sort by episode index to ensure proper plotting order
                sorted_episodes = sorted(episode_welfords.items())

                if num_episodes:
                    sorted_episodes = sorted_episodes[:num_episodes]

                mean_returns = [w.get_curr_mean_variance()[0] for _, w in sorted_episodes]
                std_returns = [np.sqrt(w.get_curr_mean_variance()[1]) for _, w in sorted_episodes]

                #x-axis corresponds to episode indices
                x_return = np.array([ep for ep, _ in sorted_episodes])

                #plot the mean return for each episode
                ax.plot(x_return, mean_returns, label=f'{agent_id} agent')

                #fill between mean pm standard deviation
                ax.fill_between(x_return,
                                np.array(mean_returns) - np.array(std_returns),
                                np.array(mean_returns) + np.array(std_returns),
                                alpha=0.2)

            ax.set_title(title)
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.show()

            #add plotting directory if desired
            plt.savefig(file_name)

    def record_return(self, agent_id: str, episode_idx: int, return_val: Union[float, int, SupportsFloat]) -> None:
        """
        Record a return value for a specific agent and episode. Uses Welford's algorithm to update mean and variance.
        :param agent_id: The identifier of the agent
        :param episode_idx: The index of the episode
        :param return_val: The return value to record
        """
        with self._lock:
            #ensure we have a Welford object for this episode in the dictionary
            if episode_idx not in self._return_history[agent_id]:
                self._return_history[agent_id][episode_idx] = Welford()

            #update the Welford object for this episode
            self._return_history[agent_id][episode_idx].update_aggr(float(return_val))

    def get_mean_std_return(self, agent_id: str, episode_idx: int) -> Tuple[Any, Any]:
        """
        Get the latest recorded return value (mean and std) for a specific agent and episode.
        :param agent_id: The identifier of the agent
        :param episode_idx: The episode index
        :return: the mean and standard deviation for the agent in the specified episode, or None if no returns have been recorded
        """
        with self._lock:
            if agent_id in self._return_history and episode_idx in self._return_history[agent_id]:
                welford = self._return_history[agent_id][episode_idx]
                mean, var = welford.get_curr_mean_variance()
                return mean, np.sqrt(var)
            else:
                return None, None

    def clear(self) -> None:
        """
        Clear the recorded metrics for all agents.
        """
        with self._lock:
            self._return_history.clear()

    @property
    def return_history(self) -> dict[str, dict[int, Welford]]:
        """
        Get the history of return values (as Welford objects) for all agents, per episode across runs.
        :return: a dictionary containing the return history for each agent (a dictionary of Welford objects for each episode).
        """
        with self._lock:
            return dict(self._return_history)