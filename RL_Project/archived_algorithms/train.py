import gymnasium as gym
import ale_py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper, EpisodicLifeEnv, FireResetEnv
from stable_baselines3.common.vec_env import VecFrameStack

def create_env(log_path):
    gym.register_envs(ale_py)
    env = gym.make("ALE/Seaquest-v5")
    env = AtariWrapper(env)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = Monitor(env, log_path)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


def setup_logging_and_monitoring(model_name):
    # Create directory structure
    log_dir = os.path.join("logs", model_name)
    model_dir = os.path.join("/content/drive/MyDrive", "models", model_name)
    eval_dir = os.path.join("logs", "eval", model_name)
    
    for d in [log_dir, model_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    # Configure the logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard", "json"])

    # # Create evaluation environment
    eval_env = create_env(os.path.join(eval_dir, "monitor"))


    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=eval_dir,
        eval_freq=25000,
        deterministic=True,
        render=False,
        n_eval_episodes = 4
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix=model_name
    )

    return eval_callback, checkpoint_callback, new_logger

def train_and_eval_dqn(model, env, total_timesteps=2000000, model_name="model"):
    # Setup logging and callbacks
    eval_callback, checkpoint_callback, new_logger = setup_logging_and_monitoring(model_name)
    
    # Set logger for model
    model.set_logger(new_logger)
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    models_dir = "/content/drive/MyDrive/models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)
    model.save(model_path)

    return model_path