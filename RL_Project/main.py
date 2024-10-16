from train import train_baseline_dqn, train_categorical_dqn
from evaluate import evaluate_dqn
from stable_baselines3 import DQN
from algorithms.categorical_dqn import CategoricalDQN
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    # create environment
    env = gym.make("ALE/Seaquest-v5")
    env = DummyVecEnv([lambda: env])
    #train baseline dqn
    # dqn_model_path = train_baseline_dqn(env)
    # dqn_model = DQN.load(dqn_model_path)
    #mean_reward, std_reward, max_reward = evaluate_dqn(dqn_model, env)

    #train categorical dqn
    c51_model_path = train_categorical_dqn(env)
    c51_model = CategoricalDQN.load(c51_model_path)
    mean_reward, std_reward, max_reward = evaluate_dqn(c51_model, env)


    print(f"Evaluation results: Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}, Max reward: {max_reward:.2f}")

    

if __name__ == "__main__":
    main()



    



