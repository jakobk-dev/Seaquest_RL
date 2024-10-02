import gymnasium as gym
from stable_baselines3 import DQN
import os

def train_model():
    # Create the environment
    env = gym.make("ALE/Seaquest-v5")


    # Create the DQN model
    model = DQN("CnnPolicy", 
    env, 
    buffer_size=5000, 
    learning_rate= 0.0001, 
    exploration_fraction = 0.2, 
    exploration_final_eps=0.05, 
    verbose=1)

    # Train the model
    model.learn(total_timesteps=100000) 

    # Define the path to the models directory
    models_dir = "/content/drive/MyDrive/models"

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(models_dir, "dqn_seaquest")
    model.save(model_path)

    print(f"Model saved to {model_path}")

    return model_path  # Return the path instead of the model object


    # # Save the model
    # model.save("models/dqn_seaquest")

    # return model
