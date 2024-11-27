import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from torch import optim

# Currently, this algorithm was not run or implmented with a graph in the paper.
class PrioritizedExperienceReplayDQN(DQN):
    """
    DQN with Prioritized Experience Replay (PER).
    Transitions with higher TD errors are replayed more frequently.
    """
    def __init__(
        self,
        policy,
        env,
        alpha=0.6,  # Prioritization exponent
        beta=0.4,   # Importance sampling weight
        beta_schedule=None,  # Schedule for beta
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

        # Initialize the Prioritized Replay Buffer
        self.alpha = alpha
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size,
            alpha=self.alpha,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Train the PER DQN model by sampling from the prioritized replay buffer.
        """
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            # Sample a batch from the prioritized replay buffer
            replay_data = self.replay_buffer.sample(batch_size, beta=self.beta)

            with torch.no_grad():
                # Compute the next actions using the policy network
                next_actions = self.policy.q_net(replay_data.next_observations).argmax(dim=1).reshape(-1, 1)

                # Compute the target Q-values using the target network
                next_q_values = self.policy.q_net_target(replay_data.next_observations).gather(1, next_actions).reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values for the sampled actions
            current_q_values = self.policy.q_net(replay_data.observations).gather(1, replay_data.actions)

            # Compute the TD error
            td_error = target_q_values - current_q_values

            # Update priorities in the replay buffer
            self.replay_buffer.update_priorities(replay_data.indices, td_error.abs().detach().cpu().numpy())

            # Compute the importance sampling weights
            weights = torch.tensor(replay_data.weights).to(self.device)

            # Compute the loss with weighted TD errors
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update the target network
            if self._n_calls % self.target_update_interval == 0:
                self.policy.q_net_target.load_state_dict(self.policy.q_net.state_dict())

        self._n_calls += gradient_steps


def create_per_dqn(env, **kwargs):
    """
    Factory function to create a Prioritized Experience Replay DQN agent.
    """
    return PrioritizedExperienceReplayDQN("CnnPolicy", env, **kwargs)
