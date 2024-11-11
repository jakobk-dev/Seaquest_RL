
import gymnasium as gym
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np

def train_dqn(model, env, total_timesteps=200000, model_name="model"):

    # Train the model
    model.learn(total_timesteps=total_timesteps) 

    # Define the path to the models directory
    models_dir = "/content/drive/MyDrive/models"

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, f"{model_name}")
    model.save(model_path)

    print(f"Model saved to {model_path}")

    return model_path 
 
