
import gymnasium as gym
from stable_baselines3 import DQN
from algorithms.categorical_dqn import create_categorical_dqn
from algorithms.noisy_dqn import create_noisy_dqn
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
    buffer_size=10000, 
    learning_rate= 1e-4, 
    exploration_fraction = 0.3, 
    exploration_final_eps=0.05, 
    verbose=1)

    # Train the model
    model.learn(total_timesteps=200000) 

    # Define the path to the models directory
    models_dir = "/content/drive/MyDrive/models"

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, "baseline_dqn_seaquest")
    model.save(model_path)

    print(f"Model saved to {model_path}")

    return model_path 

def train_categorical_dqn(env):
  # the number of atoms and value range v_min to v_max is crucial
  # paper recommendations for atom to range ratios 
  # aim for ~0.4 to 0.5 atoms per unit of value range
  # range (-10 to 10) 51 atoms
  # range (-50 to 50) 51 atoms
  # range (-100 to 100) 81 atoms 
  model = create_categorical_dqn(env, n_atoms=81, v_min=-100, v_max=100, verbose=1)


  # Train the model
  model.learn(total_timesteps=200000) 

  # Define the path to the models directory
  models_dir = "/content/drive/MyDrive/models"

  # Ensure the models directory exists
  os.makedirs(models_dir, exist_ok=True)

  # Save the model
  model_path = os.path.join(models_dir, "c51_dqn_seaquest")
  model.save(model_path)

  print(f"Model saved to {model_path}")

  return model_path  

def train_noisy_dqn(env):

  model = create_noisy_dqn(env)

  # Train the model
  model.learn(total_timesteps=200000) 

  # Define the path to the models directory
  models_dir = "/content/drive/MyDrive/models"

  # Ensure the models directory exists
  os.makedirs(models_dir, exist_ok=True)

  # Save model
  model_path = os.path.join(models_dir, "noisy_dqn_seaquest")
  model.save(model_path)

  print(f"Model saved to {model_path}")

  return model_path 


 
