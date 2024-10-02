from train import train_model
from evaluate import evaluate_model
from stable_baselines3 import DQN


if __name__ == "__main__":
    # Train the model
    model_path = train_model()

    # get/load model
    model = DQN.load(model_path)

    # Evaluate the model
    evaluate_model(model)



