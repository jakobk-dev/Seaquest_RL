
import gymnasium as gym
from stable_baselines3 import DQN
from algorithms.categorical_dqn import create_categorical_dqn
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np

def train_baseline_dqn(env):

    # Create the DQN model
    model = DQN("CnnPolicy", 
    env, 
    buffer_size=5000, 
    learning_rate= 0.0001, 
    exploration_fraction = 0.2, 
    exploration_final_eps=0.05, 
    verbose=1)

    # Train the model
    model.learn(total_timesteps=60000) 

    # Define the path to the models directory
    models_dir = "/content/drive/MyDrive/models"

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(models_dir, "baseline_dqn_seaquest")
    model.save(model_path)

    print(f"Model saved to {model_path}")

    return model_path  # Return the path of trained model

def train_categorical_dqn(env):

  model = create_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10, verbose=1)


  # Train the model
  model.learn(total_timesteps=60000) 

  # Define the path to the models directory
  models_dir = "/content/drive/MyDrive/models"

  # Ensure the models directory exists
  os.makedirs(models_dir, exist_ok=True)

  # Save the model
  model_path = os.path.join(models_dir, "c51_dqn_seaquest")
  model.save(model_path)

  print(f"Model saved to {model_path}")

  return model_path  # Return the path of trained model


 
