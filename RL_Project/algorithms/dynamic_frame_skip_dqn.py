import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import optim
import numpy as np

class DynamicFrameSkipDQN(DQN):
    """
    Dynamic Frame Skip DQN modifies the interaction with the environment to 
    adjust the frame skip dynamically based on performance.
    """
    def __init__(
        self,
        policy,
        env,
        min_frame_skip=2,
        max_frame_skip=8,
        frame_skip_adjustment_factor=0.05,
        evaluation_window=1000,
        learning_rate=1e-4,
        buffer_size=30000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
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

        # Dynamic frame skip parameters
        self.min_frame_skip = min_frame_skip
        self.max_frame_skip = max_frame_skip
        self.frame_skip_adjustment_factor = frame_skip_adjustment_factor
        self.evaluation_window = evaluation_window

        self.current_frame_skip = min_frame_skip
        self.cumulative_rewards = []
        self.previous_avg_reward = None

    def _dynamic_frame_skip_update(self, reward):
        """
        Adjust frame skip dynamically based on reward performance.
        """
        self.cumulative_rewards.append(reward)

        # Only evaluate every `evaluation_window` steps
        if len(self.cumulative_rewards) >= self.evaluation_window:
            avg_reward = np.mean(self.cumulative_rewards[-self.evaluation_window:])
            if self.previous_avg_reward is not None:
                if avg_reward > self.previous_avg_reward:
                    self.current_frame_skip = min(
                        self.current_frame_skip + 1, self.max_frame_skip
                    )
                else:
                    self.current_frame_skip = max(
                        self.current_frame_skip - 1, self.min_frame_skip
                    )
            self.previous_avg_reward = avg_reward

    def collect_rollouts(self, env, callback, train_freq, replay_buffer, action_noise=None, learning_starts=0):
        """
        Modified to include dynamic frame skipping during rollouts.
        """
        n_steps = 0
        action_noise = self.policy.action_noise if action_noise is None else action_noise

        while n_steps < train_freq:
            # Reset environment if done
            if self._last_obs is None:
                self._last_obs = env.reset()

            # Predict next action
            with torch.no_grad():
                action, buffer_action = self.predict(self._last_obs, deterministic=False)

            # Take action with dynamic frame skipping
            cumulative_reward = 0.0
            for _ in range(self.current_frame_skip):
                new_obs, reward, done, info = env.step(action)
                cumulative_reward += reward
                if done:
                    break

            # Update reward for frame skipping logic
            self._dynamic_frame_skip_update(cumulative_reward)

            # Store transition in replay buffer
            self._store_transition(self._last_obs, buffer_action, cumulative_reward, new_obs, done)
            self._last_obs = new_obs

            if done:
                self._last_obs = None  # Force env reset

            n_steps += 1

        return n_steps

def create_dynamic_frame_skip_dqn(env, **kwargs):
    """
    Factory function to create a Dynamic Frame Skip DQN agent.
    """
    return DynamicFrameSkipDQN("CnnPolicy", env, **kwargs)
