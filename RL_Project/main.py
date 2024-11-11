from train import train_dqn
from evaluate import evaluate_dqn
from stable_baselines3 import DQN
from algorithms.categorical_dqn import CategoricalDQN
from algorithms.noisy_dqn import NoisyDQN
from algorithms.noisy_categorical_dqn import NoisyCategoricalDQN
from algorithms.categorical_dqn import create_categorical_dqn
from algorithms.noisy_dqn import create_noisy_dqn
from algorithms.noisy_categorical_dqn import create_noisy_categorical_dqn
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    # create environment
    env = gym.make("ALE/Seaquest-v5")
    env = DummyVecEnv([lambda: env])
    #train baseline dqn
    # Create the DQN model
    baseline_model = DQN("CnnPolicy", env, buffer_size=30000, learning_rate= 1e-4, exploration_fraction = 0.3, exploration_final_eps=0.05, verbose=1)
    dqn_model_path = train_dqn(baseline_model, env, "baseline_dqn_model")
    dqn_model = DQN.load(dqn_model_path)
    mean_reward, std_reward, max_reward = evaluate_dqn(dqn_model, env)
    print(f"Evaluation results for baseline: Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}, Max reward: {max_reward:.2f}")

    #train categorical dqn
    c51_model = create_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10, verbose=1)
    c51_model_path = train_dqn(c51_model, env, "c51_dqn_model")
    c51_model = CategoricalDQN.load(c51_model_path)
    mean_reward, std_reward, max_reward = evaluate_dqn(c51_model, env)
    print(f"Evaluation results for c51: Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}, Max reward: {max_reward:.2f}")

    #train noisy dqn
    noisy_model = create_noisy_dqn(env)
    noisy_model_path = train_dqn(noisy_model, env, "noisy_dqn_model")
    noisy_model = NoisyDQN.load(noisy_model_path)
    mean_reward, std_reward, max_reward = evaluate_dqn(noisy_model, env)
    print(f"Evaluation results for noisy: Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}, Max reward: {max_reward:.2f}")

    # train noisy/categorical dqn variant
    noisy_categorical_model = create_noisy_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10, verbose=1)
    noisy_categorical_model_path = train_dqn(noisy_categorical_model, env, "noisy_categorical_variant_model")
    noisy_categorical_model = NoisyCategoricalDQN.load(noisy_categorical_model_path)
    mean_reward, std_reward, max_reward = evaluate_dqn(noisy_categorical_model, env)
    print(f"Evaluation results for noisy/c51 variant: Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}, Max reward: {max_reward:.2f}")


if __name__ == "__main__":
    main()



    



