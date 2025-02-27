import gymnasium as gym
from rl_cliff.agents.abstractagent import AbstractAgent
from rl_cliff.util.metricstracker import MetricsTracker


def evaluate_target_policy(env: gym.Env, agent: AbstractAgent, tracker: MetricsTracker,
                           episode_idx: int) -> float:
    """
    Evaluates the target policy for a single episode and records the return.

    :param env: gym environment
    :param agent: an agent with a target policy
    :param tracker: used to track the episode returns
    :param episode_idx: index of the current episode
    :return: total return for the episode
    """
    episode_return = 0
    obs, info = env.reset()

    #сan be extended to return the avg+std return for multiple episodes but for the purposes of the
    #assignment we are interested in one evaluation episode since we directly want to compare the
    #behavioral and target policy
    while True:
        action = agent.target_policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        #сonsider falling from the Cliff as a terminal state
        if reward == -100:
            terminated = True

        episode_return += reward
        obs = next_obs

        if terminated or truncated:
            tracker.record_return(agent_id=agent.agent_type, return_val=episode_return, episode_idx=episode_idx)
            break

    return episode_return