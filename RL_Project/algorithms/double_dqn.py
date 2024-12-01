import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from torch import optim
import numpy as np


class ImprovedLongDoubleDQN(DQN):
    """
    Double DQN with improvements: Noisy exploration and optimized hyperparameters.
    """
    def __init__(
        self,
        policy,
        env,
        learning_rate=5e-4,  # Increased learning rate
        buffer_size=100000,  # Larger replay buffer
        batch_size=128,  # Larger batch size for better gradient estimates
        tau=0.005,  # Polyak averaging for smoother updates
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,  # Frequent target network updates
        exploration_fraction=0.2,  # Faster decay of exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,  # More exploitation in later stages
        max_grad_norm=10,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=1,
        device="auto",
        _init_setup_model=True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 128) -> None:
        """
        Override the DQN train method to implement Double Q-Learning with noisy exploration.
        """
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            # Sample from the replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Double Q-Learning: Use policy for action selection, target for evaluation
                next_actions = self.policy.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_values = self.policy.q_net_target(replay_data.next_observations).gather(1, next_actions)

                # Compute TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values
            current_q_values = self.policy.q_net(replay_data.observations).gather(1, replay_data.actions)

            # Compute loss
            td_errors = target_q_values - current_q_values
            loss = td_errors.pow(2).mean()

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Soft update for target network
            for target_param, param in zip(self.policy.q_net_target.parameters(), self.policy.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self._n_calls += gradient_steps


def create_double_dqn(env, **kwargs):
    """
    Factory function to create an improved Double DQN agent.
    """
    policy_kwargs = dict(
        net_arch=[256, 256],  # Deeper network for better representation
        activation_fn=torch.nn.ReLU,
    )
    kwargs['policy_kwargs'] = policy_kwargs
    return ImprovedLongDoubleDQN("CnnPolicy", env, **kwargs)
