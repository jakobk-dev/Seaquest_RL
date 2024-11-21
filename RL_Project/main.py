from train import train_and_eval_dqn, create_env
from stable_baselines3 import DQN
from algorithms.categorical_dqn import create_categorical_dqn
from algorithms.noisy_dqn import create_noisy_dqn
from algorithms.noisy_categorical_dqn import create_noisy_categorical_dqn
import os
from plotting import plot_metrics

def main():
    base_log_path = "/content/drive/MyDrive/logs/monitor"
    os.makedirs(base_log_path, exist_ok=True)
    env = create_env(base_log_path)
    
    #train baseline dqn
    # Create the DQN model
    # baseline_model = DQN("CnnPolicy", env, buffer_size=30000, learning_rate= 1e-4, batch_size=32, exploration_fraction = 0.3, exploration_final_eps=0.05,tensorboard_log="/content/drive/MyDrive/logs/tensorboard",verbose=1)
    # dqn_model_path = train_and_eval_dqn(baseline_model, env,total_timesteps = 200000, model_name= "baseline_dqn_model")
    # csv_path = "/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/baseline_dqn_model/progress.csv"
    # plot_metrics(csv_path, plot_title ='Baseline DQN Training and Evaluation Analysis', plot_file_name = 'baseline_dqn_results_graph.png')
   
    # #train categorical dqn
    # c51_model = create_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10, batch_size = 32, exploration_final_eps=0.05,tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    # c51_model_path = train_and_eval_dqn(c51_model, env, total_timesteps = 200000, model_name = "c51_dqn_model_1") 
    # csv_path = "/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/c51_dqn_model_1/progress.csv"
    # plot_metrics(csv_path, plot_title ='C51 DQN Training and Evaluation Analysis', plot_file_name = 'c51_dqn_1_results_graph.png')

    # c51_model = create_categorical_dqn(env, n_atoms=51, v_min=-50, v_max=50, batch_size = 32, exploration_final_eps=0.05,tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    # c51_model_path = train_and_eval_dqn(c51_model, env, total_timesteps = 200000, model_name = "c51_dqn_model_2") 
    # csv_path = "/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/c51_dqn_model_2/progress.csv"
    # plot_metrics(csv_path, plot_title ='C51 DQN Training and Evaluation Analysis', plot_file_name = 'c51_dqn_2_results_graph.png')

    # c51_model = create_categorical_dqn(env, n_atoms=81, v_min=-100, v_max=100, batch_size = 32, exploration_final_eps=0.05,tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    # c51_model_path = train_and_eval_dqn(c51_model, env, total_timesteps = 200000, model_name = "c51_dqn_model_3") 
    # csv_path = "/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/c51_dqn_model_3/progress.csv"
    # plot_metrics(csv_path, plot_title ='C51 DQN Training and Evaluation Analysis', plot_file_name = 'c51_dqn_1_results_graph.png')

   
    # #train noisy dqn
    #std_init set to 0.1
    # noisy_model = create_noisy_dqn(env, batch_size = 128, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    # noisy_model_path = train_and_eval_dqn(noisy_model, env, total_timesteps = 200000, model_name="noisy_dqn_model_1")
    # csv_path = "/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/noisy_dqn_model_1/progress.csv"
    # plot_metrics(csv_path, plot_title ='Noisy DQN Training and Evaluation Analysis', plot_file_name = 'noisy_dqn_1_results_graph.png')

    #std_init set to 0.5
    noisy_model = create_noisy_dqn(env, batch_size = 128, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    noisy_model_path = train_and_eval_dqn(noisy_model, env, total_timesteps = 200000, model_name="noisy_dqn_model_2")
    csv_path = "/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/noisy_dqn_model_2/progress.csv"
    plot_metrics(csv_path, plot_title ='Noisy DQN Training and Evaluation Analysis', plot_file_name = 'noisy_dqn_2_results_graph.png')

    #std_init set to 1.0
    # noisy_model = create_noisy_dqn(env, batch_size = 128, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    # noisy_model_path = train_and_eval_dqn(noisy_model, env, total_timesteps = 200000, model_name="noisy_dqn_model_3")
    # csv_path = "/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/noisy_dqn_model_3/progress.csv"
    # plot_metrics(csv_path, plot_title ='Noisy DQN Training and Evaluation Analysis', plot_file_name = 'noisy_dqn_3_results_graph.png')

    
    # # train noisy/categorical dqn variant
    # noisy_categorical_model = create_noisy_categorical_dqn(env, n_atoms=51, v_min=-10, v_max=10, batch_size=32, tensorboard_log="/content/drive/MyDrive/logs/tensorboard", verbose=1)
    # noisy_categorical_model_path = train_and_eval_dqn(noisy_categorical_model, env, total_timesteps = 200000,model_name="noisy_categorical_variant_model")
    # csv_path = "/content/drive/MyDrive/Seaquest_RL/RL_Project/logs/noisy_categorical_variant_model/progress.csv"
    # plot_metrics(csv_path, plot_title ='Categorical/Noisy DQN Training and Evaluation Analysis', plot_file_name = 'c51_noisy_dqn_results_graph.png')

   

if __name__ == "__main__":
    main()



    



