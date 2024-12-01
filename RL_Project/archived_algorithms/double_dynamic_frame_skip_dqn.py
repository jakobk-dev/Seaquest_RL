import gym
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy

class DoubleDynamicFrameSkipDQN(DQN):
    """
    DQN implementation that combines Double DQN to reduce overestimation bias
    and Dynamic Frame Skipping which adjusts the frame skip dynamically during training based on the TD error.
    """
    def __init__(
        self,
        policy,
        env,
        initial_frame_skip=1,
        max_frame_skip=4,
        frame_skip_increment=1,
        td_error_threshold=1.0,
        learning_rate=1e-4,
        buffer_size=50000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.1,
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
        self.current_frame_skip = initial_frame_skip
        self.max_frame_skip = max_frame_skip
        self.frame_skip_increment = frame_skip_increment
        self.td_error_threshold = td_error_threshold

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Override the DQN train method to implement Double Q-Learning updates and Dynamic Frame Skipping.
        """
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Double DQN: Compute next actions using the policy network (action selection)
                next_actions = self.policy.q_net(replay_data.next_observations).argmax(dim=1).reshape(-1, 1)

                # Evaluate the selected actions using the target network
                next_q_values = self.policy.q_net_target(replay_data.next_observations).gather(1, next_actions).reshape(-1, 1)

                # Compute the TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.policy.q_net(replay_data.observations).gather(1, replay_data.actions)

            # Compute the TD error
            td_errors = target_q_values - current_q_values
            td_error_mean = td_errors.abs().mean().item()

            # Compute the loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update target network
            if self._n_calls % self.target_update_interval == 0:
                self.policy.q_net_target.load_state_dict(self.policy.q_net.state_dict())

            # Adjust frame skip based on TD error
            if td_error_mean > self.td_error_threshold and self.current_frame_skip < self.max_frame_skip:
                self.current_frame_skip += self.frame_skip_increment
            elif td_error_mean < self.td_error_threshold and self.current_frame_skip > 1:
                self.current_frame_skip -= self.frame_skip_increment

            # Update the environment's frame skip
            if hasattr(self.env, 'set_frame_skip'):
                self.env.set_frame_skip(self.current_frame_skip)
            elif hasattr(self.env.envs[0], 'set_frame_skip'):
                # For vectorized environments
                for env in self.env.envs:
                    env.set_frame_skip(self.current_frame_skip)

        self._n_calls += gradient_steps

def create_double_dynamic_frame_skip_dqn(env, **kwargs):
    """
    Factory function to create a Double DQN agent with Dynamic Frame Skipping.
    """
    return DoubleDynamicFrameSkipDQN("CnnPolicy", env, **kwargs)
