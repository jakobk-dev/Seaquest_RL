import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from torch import optim
import numpy as np

class RainbowDQN(DQN):
    """
    Rainbow DQN combines Double Q-Learning, Prioritized Replay, Dueling Architecture,
    Multi-step Learning, Noisy Networks, and Distributional RL.
    Currently, this algorithm was not run or implmented with a graph in the paper.
    """
    def __init__(
        self,
        policy,
        env,
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        alpha=0.6,  # PER: prioritization exponent
        beta=0.4,   # PER: importance sampling weight
        multi_step=3,  # Multi-step returns
        learning_rate=1e-4,
        buffer_size=30000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.0,  # Not used with Noisy Nets
        exploration_initial_eps=0.0,  # Not used with Noisy Nets
        exploration_final_eps=0.0,  # Not used with Noisy Nets
        max_grad_norm=10,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=1,
        device="auto",
        _init_setup_model=True,
    ):
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.update({
            "n_atoms": n_atoms,
            "v_min": v_min,
            "v_max": v_max,
            "optimizer_class": optim.Adam,
            "optimizer_kwargs": {"eps": 0.01 / batch_size},
        })

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

        # Rainbow-specific parameters
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.alpha = alpha
        self.beta = beta
        self.multi_step = multi_step

        # Initialize Prioritized Experience Replay
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size,
            alpha=alpha,
            beta_initial=beta,
            beta_schedule=None,
            n_step=multi_step,
            gamma=gamma,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Train the Rainbow DQN model using Prioritized Replay and Multi-step Learning.
        """
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            # Sample from prioritized replay buffer
            replay_data = self.replay_buffer.sample(batch_size, beta=self.beta_schedule.value(self._n_calls))

            with torch.no_grad():
                # Distributional Bellman Update
                self.policy.q_net_target.reset_noise()
                next_distributions = self.policy.q_net_target(replay_data.next_observations)

                next_q_values = (next_distributions * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
                next_actions = next_q_values.argmax(dim=1).reshape(-1, 1)

                target_distributions = next_distributions[
                    torch.arange(batch_size, device=self.device), next_actions.squeeze()
                ]

                Tz = replay_data.rewards.to(self.device) + (1 - replay_data.dones.to(self.device)) * self.gamma * self.support
                Tz = Tz.clamp(min=self.v_min, max=self.v_max)

                b = (Tz - self.v_min) / self.delta_z
                l, u = b.floor().long(), b.ceil().long()
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.n_atoms - 1)) * (l == u)] += 1

                m = torch.zeros_like(target_distributions)
                offset = (
                    torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size)
                    .unsqueeze(1)
                    .expand(batch_size, self.n_atoms)
                    .long()
                    .to(self.device)
                )
                m.view(-1).index_add_(0, (l + offset).view(-1), (target_distributions * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (target_distributions * (b - l.float())).view(-1))

            self.policy.q_net.reset_noise()

            current_distributions = self.policy.q_net(replay_data.observations)
            actions = replay_data.actions.long().flatten()
            current_distribution = current_distributions[
                torch.arange(batch_size, device=self.device), actions
            ]

            loss = -(m * torch.log(current_distribution + 1e-8)).sum(dim=1).mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            self.replay_buffer.update_priorities(replay_data.indices, loss.detach().cpu().numpy())
            if self._n_calls % self.target_update_interval == 0:
                self.policy.q_net_target.load_state_dict(self.policy.q_net.state_dict())

        self._n_calls += gradient_steps


def create_rainbow_dqn(env, **kwargs):
    """
    Factory function to create a Rainbow DQN agent.
    """
    return RainbowDQN("CnnPolicy", env, **kwargs)
