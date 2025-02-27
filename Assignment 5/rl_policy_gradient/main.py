import gymnasium as gym
from loguru import logger
import mujoco
from stable_baselines3 import PPO
from rl_policy_gradient.agents.agentfactory import AgentFactory
from rl_policy_gradient.util.metricstracker import MetricsTracker
from rl_policy_gradient.util.metricstrackercallback import MetricsTrackerCallback


def env_interaction(env_str: str,
                    agent_type: str,
                    num_episodes: int,
                    tracker: MetricsTracker,
                    learning_rate_actor: float,
                    learning_rate_critic: float) -> None:
    """
    Train RL algorithms and track performance.
    :param env_str: environment name as a string
    :param agent_type: type of agent to be trained
    :param num_episodes: number of training episodes
    :param tracker: tracks performance of behavioral policy
    :param learning_rate_actor: learning rate for policy approximation
    :param learning_rate_critic: learning rate for value function approximation
    """
    logger.info(f"Training {agent_type} on {env_str} for {num_episodes} episodes.")

    env = gym.make(env_str, render_mode='rgb_array')
    agent = AgentFactory.create_agent(agent_type, env=env,
                                      lr_actor=learning_rate_actor,
                                      lr_critic=learning_rate_critic)

    for episode in range(num_episodes):
        episode_return = 0
        obs, info = env.reset()

        #start episode loop
        while True:
            action = agent.policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, next_obs, terminated or truncated)

            episode_return += reward
            obs = next_obs

            if terminated or truncated:
                #record episodic return for the policy
                logger.info(f"Episode finished, {episode_return}")
                tracker.record_metric("return", agent_id=agent_type, episode_idx=episode,
                                      value=episode_return)
                break

    env.close()


def train_ppo(env_str: str, tracker: MetricsTracker) -> None:
    """
    Train your PPO algorithm here and track performance using MetricsTracker.
    Use the MetricsTrackerCallback when running the learn method
    :param env_str: environment name as a string
    :param tracker: metric tracker
    """
    logger.info(f"Training PPO on {env_str}")

    env = gym.make(env_str, render_mode='rgb_array')

    metrics_tracker_callback = MetricsTrackerCallback(tracker)

    #set up PPO model with desired parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        verbose=1
    )

    #train model with the MetricsTrackerCallback
    model.learn(
        total_timesteps=10000,
        callback=metrics_tracker_callback
    )

    env.close()


if __name__ == "__main__":
    metric_tracker = MetricsTracker()

    num_runs = 5
    n_episodes = 500
    for _ in range(num_runs):
        #example training calls for each algorithm
        env_interaction("InvertedPendulum-v4", agent_type="ACTOR-CRITIC-AGENT", num_episodes=n_episodes,
                        tracker=metric_tracker, learning_rate_actor=0.001, learning_rate_critic=0.01)
        env_interaction("InvertedPendulum-v4", agent_type="REINFORCE-AGENT", num_episodes=n_episodes,
                        tracker=metric_tracker, learning_rate_actor=0.001, learning_rate_critic=0.01)
        train_ppo("CartPole-v1", tracker=metric_tracker)

    #plot the metric
    metric_tracker.plot_metric("return", "target_return.png", num_episodes=n_episodes,
                               x_axis_label="Episodes", y_axis_label="Average Return",
                               title="Return vs Episodes for RL Agents")

    logger.info("Training completed. Plot saved as 'target_return.png'.")