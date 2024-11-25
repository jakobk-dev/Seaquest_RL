import argparse
from train import train_and_eval_dqn, create_env
from stable_baselines3 import DQN
from algorithms.categorical_dqn import create_categorical_dqn
from algorithms.noisy_dqn import create_noisy_dqn
from algorithms.noisy_categorical_dqn import create_noisy_categorical_dqn
from algorithms.dynamic_frame_skip_dqn import create_dynamic_frame_skip_dqn
# from algorithms.rainbow_dqn import create_rainbow_dqn
from algorithms.double_dqn import create_double_dqn
# from algorithms.per_dqn import create_per_dqn
from algorithms.dynamic_action_repetition_dqn import create_dynamic_action_repetition_dqn
import os
from plotting import plot_metrics

def main(dqn_type):
    base_log_path = "/content/drive/MyDrive/logs/monitor"
    os.makedirs(base_log_path, exist_ok=True)
    env = create_env(base_log_path)
    
    model = None
    model_name = None
    csv_path = None
    plot_title = None
    plot_file_name = None

    if dqn_type == "baseline":
        model = DQN("CnnPolicy", env, buffer_size=30000, learning_rate=1e-4, batch_size=32, exploration_fraction=0.3, exploration_final_eps=0.05, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
        model_name = "baseline_dqn_model"
        plot_title = "Baseline DQN Training and Evaluation Analysis"
        plot_file_name = "baseline_dqn_results_graph.png"
    elif dqn_type == "categorical":
        model = create_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10, batch_size=32, exploration_final_eps=0.05, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
        model_name = "categorical_dqn_model"
        plot_title = "Categorical DQN Training and Evaluation Analysis"
        plot_file_name = "categorical_dqn_results_graph.png"
    elif dqn_type == "noisy":
        model = create_noisy_dqn(env, batch_size=128, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
        model_name = "noisy_dqn_model"
        plot_title = "Noisy DQN Training and Evaluation Analysis"
        plot_file_name = "noisy_dqn_results_graph.png"
    elif dqn_type == "noisy_categorical":
        model = create_noisy_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10, batch_size=32, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
        model_name = "noisy_categorical_dqn_model"
        plot_title = "Noisy Categorical DQN Training and Evaluation Analysis"
        plot_file_name = "noisy_categorical_dqn_results_graph.png"
    elif dqn_type == "dfdqn":
        model = create_dynamic_frame_skip_dqn(env, batch_size=32, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
        model_name = "dfdqn_model"
        plot_title = "Dynamic Frame Skip DQN Training and Evaluation Analysis"
        plot_file_name = "dfdqn_results_graph.png"
    # elif dqn_type == "rainbow":
    #     model = create_rainbow_dqn(env, n_atoms=51, v_min=-10, v_max=10, batch_size=32, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    #     model_name = "rainbow_dqn_model"
    #     plot_title = "Rainbow DQN Training and Evaluation Analysis"
    #     plot_file_name = "rainbow_dqn_results_graph.png"
    elif dqn_type == "double":
        model = create_double_dqn(env, batch_size=32, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
        model_name = "double_dqn_model"
        plot_title = "Double Q-Learning DQN Training and Evaluation Analysis"
        plot_file_name = "double_dqn_results_graph.png"
    # elif dqn_type == "per":
    #     model = create_per_dqn(env, batch_size=32, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    #     model_name = "per_dqn_model"
    #     plot_title = "Prioritized Experience Replay DQN Training and Evaluation Analysis"
    #     plot_file_name = "per_dqn_results_graph.png"
    elif dqn_type == "dar":
        model = create_dynamic_action_repetition_dqn(env, batch_size=32, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
        model_name = "dar_dqn_model"
        plot_title = "Dynamic Action Repetition DQN Training and Evaluation Analysis"
        plot_file_name = "dar_dqn_results_graph.png"
    else:
        print(f"Invalid DQN type: {dqn_type}")
        print("Type of DQN to run \nbaseline, categorical, noisy, noisy_categorical, dfdqn, rainbow, double, per, or dar")
        return

    model_path = train_and_eval_dqn(model, env, total_timesteps=200000, model_name=model_name) # Increase timesteps as needed
    csv_path = f"/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/{model_name}/progress.csv"
    plot_metrics(csv_path, plot_title=plot_title, plot_file_name=plot_file_name)

if __name__ == "__main__":
    print("Type of DQN to run \nbaseline, categorical, noisy, noisy_categorical, dfdqn, rainbow, double, per, or dar")
    parser = argparse.ArgumentParser()
    parser.add_argument("dqn_type", type=str)
    args = parser.parse_args()
    main(args.dqn_type)
