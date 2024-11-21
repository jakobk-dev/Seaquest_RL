import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True

def plot_metrics(csv_path, save_dir='plots', plot_title = 'Default', plot_file_name = 'Default'):
    """
    Plot training and evaluation metrics from CSV file
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(plot_title, fontsize=16) 
    ax1.plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], 
                 label='Training Reward', color='blue', alpha=0.6)
        
    eval_mask = df['eval/mean_reward'].notna()
    if eval_mask.any():
            ax1.scatter(df.loc[eval_mask, 'time/total_timesteps'],
                       df.loc[eval_mask, 'eval/mean_reward'],
                       label='Evaluation Reward', color='red', s=100)
        
    ax1.set_title('Rewards over Training')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Mean Reward')
    ax1.legend()
    ax1.grid(True)
        
    ax2.plot(df['time/total_timesteps'], df['rollout/ep_len_mean'],
                 label='Training Episode Length', color='blue', alpha=0.6)
        
    if eval_mask.any():
            ax2.scatter(df.loc[eval_mask, 'time/total_timesteps'],
                       df.loc[eval_mask, 'eval/mean_ep_length'],
                       label='Evaluation Episode Length', color='red', s=100)
        
    ax2.set_title('Episode Lengths over Training')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Mean Episode Length')
    ax2.legend()
    ax2.grid(True)
        
    plt.tight_layout()
        
    save_path = os.path.join(save_dir, plot_file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
        