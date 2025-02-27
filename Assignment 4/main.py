import copy
import gymnasium as gym
from loguru import logger
from stable_baselines3 import DQN
from rl_cliff.agents.agentfactory import AgentFactory
from rl_cliff.util.earlystopcallback import EarlyStopOnNegativeRewardCallback
from rl_cliff.util.evaluation import evaluate_target_policy
from rl_cliff.util.metricstracker import MetricsTracker
from rl_cliff.util.metricstrackercallback import MetricsTrackerCallback


def env_interaction(env_str: str, agent_type: str, num_episodes: int,
                    behavioral_tracker: MetricsTracker, target_tracker: MetricsTracker,
                    epsilon: float, learning_rate: float) -> None:
    """
    Train tabular algorithms and track performance.
    :param env_str: environment name as a string.
    :param agent_type: type of agent to be trained.
    :param num_episodes: number of training episodes.
    :param behavioral_tracker: tracks performance of behavioral policy.
    :param target_tracker: tracks performance of target policy.
    :param epsilon: exploration rate.
    :param learning_rate: learning rate for the agent.
    """
    logger.info(f"Training {agent_type} on {env_str} for {num_episodes} episodes.")

    # Create environment and a copy for evaluation purposes
    env = gym.make(env_str, render_mode='rgb_array', max_episode_steps=1_00)
    eval_env = copy.deepcopy(env)

    agent = AgentFactory.create_agent(agent_type, env=env, eps=epsilon, lr=learning_rate)

    for episode in range(num_episodes):
        episode_return = 0
        obs, info = env.reset()

        # Start episode loop
        while True:
            action = agent.behavior_policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Consider falling from the Cliff as a terminal state
            if reward == -100:
                terminated = True

            agent.update((obs, action, reward, next_obs))

            episode_return += reward
            obs = next_obs

            if terminated or truncated:
                # Record episodic return for the behavioral policy
                behavioral_tracker.record_return(agent_id=agent.agent_type, episode_idx=episode,
                                                 return_val=episode_return)

                # Evaluate target policy and record its return
                evaluate_target_policy(eval_env, agent, target_tracker, episode_idx=episode)
                break

    # Close environments after interaction
    env.close()
    eval_env.close()


def train_dqn(env_str: str, behavioral_tracker: MetricsTracker, target_tracker: MetricsTracker) -> None:
    """
    Train your DQN algorithm here and track performance using MetricsTracker.
    Use the MetricsTrackerCallback and EarlyStopOnNegativeRewardCallback when running the learn method.
    :param env_str: environment name as a string.
    :param behavioral_tracker: tracks performance of behavioral policy.
    :param target_tracker: tracks performance of target policy.
    """
    logger.info(f"Training DQN on {env_str}")

    env = gym.make(env_str, render_mode='rgb_array', max_episode_steps=1_00)
    eval_env = copy.deepcopy(env)

    model = DQN("MlpPolicy", env, verbose=0, learning_rate=0.001, target_update_interval=10, buffer_size=1000)

    #define the callbacks
    metrics_tracker_callback = MetricsTrackerCallback(behavioral_tracker, target_tracker, eval_env)
    early_stop_callback = EarlyStopOnNegativeRewardCallback(stop_reward=-100)

    model.learn(total_timesteps=100000, callback=[early_stop_callback, metrics_tracker_callback])
    model.save("dqn_model")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    tracker_behavior = MetricsTracker()
    tracker_target = MetricsTracker()

    num_runs = 15
    for _ in range(num_runs):
        # env_interaction("CliffWalking-v0", "SARSA", 7000, tracker_behavior, tracker_target, 0.1, 0.01)
        # env_interaction("CliffWalking-v0", "Q-LEARNING", 7000, tracker_behavior, tracker_target, 0.1, 0.01)
        # env_interaction("CliffWalking-v0", "DOUBLE-Q-LEARNING", 7000, tracker_behavior, tracker_target, 0.1, 0.01)
        train_dqn("CliffWalking-v0", tracker_behavior, tracker_target)

    tracker_behavior.plot("behavior_return.png")
    tracker_target.plot("target_return.png")