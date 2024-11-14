from train import train_and_eval_dqn
from stable_baselines3 import DQN
from algorithms.categorical_dqn import CategoricalDQN
from algorithms.noisy_dqn import NoisyDQN
from algorithms.noisy_categorical_dqn import NoisyCategoricalDQN
from algorithms.categorical_dqn import create_categorical_dqn
from algorithms.noisy_dqn import create_noisy_dqn
from algorithms.noisy_categorical_dqn import create_noisy_categorical_dqn
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper, EpisodicLifeEnv, FireResetEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage
import os

def create_env(log_path):
    env = gym.make("ALE/Seaquest-v5")
    env = AtariWrapper(env)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = Monitor(env, log_path)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env

def main():
    base_log_path = "/content/drive/MyDrive/logs/monitor"
    os.makedirs(base_log_path, exist_ok=True)
    env = create_env(base_log_path)

    #train baseline dqn
    # Create the DQN model
    # baseline_model = DQN("CnnPolicy", env, buffer_size=30000, learning_rate= 1e-4, exploration_fraction = 0.3, exploration_final_eps=0.05,tensorboard_log="/content/drive/MyDrive/logs/tensorboard",verbose=1)
    # dqn_model_path = train_and_eval_dqn(baseline_model, env,total_timesteps = 200000, model_name= "baseline_dqn_model")
   
    # #train categorical dqn
    # c51_model = create_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10,exploration_final_eps=0.05,tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    # c51_model_path = train_and_eval_dqn(c51_model, env, total_timesteps = 200000, model_name = "c51_dqn_model")
   
    # #train noisy dqn
    # noisy_model = create_noisy_dqn(env, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    # noisy_model_path = train_and_eval_dqn(noisy_model, env, total_timesteps = 200000, model_name="noisy_dqn_model")
    
    # # train noisy/categorical dqn variant
    noisy_categorical_model = create_noisy_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    noisy_categorical_model_path = train_and_eval_dqn(noisy_categorical_model, env, total_timesteps = 200000,model_name="noisy_categorical_variant_model")
   

if __name__ == "__main__":
    main()



    



