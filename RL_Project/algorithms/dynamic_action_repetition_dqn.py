import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from torch import optim
import numpy as np


class DynamicActionRepetitionDQN(DQN):
    """
    Dynamic Action Repetition DQN modifies the interaction with the environment
    to dynamically adjust the number of times an action is repeated during training.
    """
    def __init__(
        self,
        policy,
        env,
        min_action_repeats=1,
        max_action_repeats=5,
        adjustment_threshold=10,
        reward_sensitivity=0.1,
        learning_rate=1e-4,
        buffer_size=30000,
        batch_size=64,
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

        # Dynamic Action Repetition Parameters
        self.min_action_repeats = min_action_repeats
        self.max_action_repeats = max_action_repeats
        self.adjustment_threshold = adjustment_threshold
        self.reward_sensitivity = reward_sensitivity

        self.current_action_repeats = min_action_repeats
        self.last_rewards = []

    def _adjust_action_repetition(self, reward):
        """
        Adjusts the number of times an action is repeated based on the reward feedback.
        """
        self.last_rewards.append(reward)
        if len(self.last_rewards) > self.adjustment_threshold:
            avg_reward = np.mean(self.last_rewards[-self.adjustment_threshold:])
            delta_reward = avg_reward - np.mean(self.last_rewards[:-self.adjustment_threshold])
            if delta_reward > self.reward_sensitivity:
                self.current_action_repeats = min(self.current_action_repeats + 1, self.max_action_repeats)
            elif delta_reward < -self.reward_sensitivity:
                self.current_action_repeats = max(self.current_action_repeats - 1, self.min_action_repeats)

    def collect_rollouts(self, env, callback, train_freq, replay_buffer, action_noise=None, learning_starts=0):
        """
        Collect rollouts with dynamic action repetition during environment interaction.
        """
        n_steps = 0
        action_noise = self.policy.action_noise if action_noise is None else action_noise

        while n_steps < train_freq:
            # Reset the environment if done
            if self._last_obs is None:
                self._last_obs = env.reset()

            # Predict the next action
            with torch.no_grad():
                action, buffer_action = self.predict(self._last_obs, deterministic=False)

            # Take the action repeatedly and accumulate rewards
            cumulative_reward = 0.0
            for _ in range(self.current_action_repeats):
                new_obs, reward, done, info = env.step(action)
                cumulative_reward += reward
                if done:
                    break

            # Adjust the action repetition based on the cumulative reward
            self._adjust_action_repetition(cumulative_reward)

            # Store the transition in the replay buffer
            self._store_transition(self._last_obs, buffer_action, cumulative_reward, new_obs, done)
            self._last_obs = new_obs

            if done:
                self._last_obs = None  # Force environment reset

            n_steps += 1

        return n_steps


def create_dynamic_action_repetition_dqn(env, **kwargs):
    """
    Factory function to create a Dynamic Action Repetition DQN agent.
    """
    return DynamicActionRepetitionDQN("CnnPolicy", env, **kwargs)
