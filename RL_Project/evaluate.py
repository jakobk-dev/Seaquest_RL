import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def evaluate_dqn(model, env, num_episodes=10):
    
    episode_rewards = []
    max_reward = float('-inf')

    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

        episode_rewards.append(episode_reward)
        max_reward = max(max_reward, episode_reward)

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    
    return mean_reward, std_reward, max_reward
