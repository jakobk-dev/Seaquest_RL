import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def evaluate_model(model, num_episodes=10):
    # Create and wrap the environment
    env = DummyVecEnv([lambda: gym.make("ALE/Seaquest-v5")])

    episode_scores = []
    max_score = float('-inf')

    for _ in range(num_episodes):
        obs = env.reset()
        episode_score = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_score += reward[0] 

        episode_scores.append(episode_score)
        max_score = max(max_score, episode_score)

    env.close()

    average_score = np.mean(episode_scores)

    print(f"Average Score: {average_score}")
    print(f"Max Score: {max_score}")
    
    return average_score, max_score
