import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import optim

class CategoricalNetwork(nn.Module):
    def __init__(self, observation_space, action_space, n_atoms=51):
        super().__init__()
        
        n_input_channels = observation_space.shape[0]
        self.n_actions = action_space.n
        self.n_atoms = n_atoms
        
        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Compute the feature size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            conv_out = self.features(sample_input)
            self.feature_size = conv_out.view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions * self.n_atoms)
        )

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        batch_size = x.shape[0]
        conv_out = self.features(x)
        flattened = conv_out.view(batch_size, -1)
        out = self.fc(flattened)
        out = out.view(batch_size, self.n_actions, self.n_atoms)
        return F.softmax(out, dim=2)

    def get_q_values(self, x):
        distribution = self.forward(x)
        return (distribution * torch.arange(self.n_atoms).float().to(distribution.device)).sum(dim=2)

class CategoricalPolicy(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_atoms=51,
        net_arch=None,
        activation_fn=nn.ReLU,
        features_extractor_class=CategoricalNetwork,
        features_extractor_kwargs=None,
        optimizer_class=optim.Adam,
        optimizer_kwargs=None,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        self.n_atoms = n_atoms
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_atoms": self.n_atoms,
        }
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self):
        return self.features_extractor_class(**self.net_args)

    def forward(self, obs, deterministic=True):
        return self._predict(obs, deterministic)

    def _predict(self, obs, deterministic=True):
        q_values = self.q_net.get_q_values(obs)
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def extract_features(self, obs):
        return obs

class CategoricalDQN(DQN):
    def __init__(
        self,
        policy,
        env,
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=50000,
        batch_size=32,
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
        create_eval_env=False,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device='auto',
        _init_setup_model=True
    ):
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs["n_atoms"] = n_atoms

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
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
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
      # Switch to train mode (this affects batch norm / dropout)
      self.policy.set_training_mode(True)

      for _ in range(gradient_steps):
          # Sample replay buffer
          replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

          with torch.no_grad():
              # Compute the next Q-values using the target network
              next_state_distributions = self.q_net_target(replay_data.next_observations)
              
              # Follow greedy policy: use the one with the highest value
              next_q_values = (next_state_distributions * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
              next_actions = next_q_values.argmax(dim=1).reshape(-1, 1)
              
              # Compute the target distribution
              target_distributions = next_state_distributions[torch.arange(batch_size), next_actions.squeeze()]
              
              # Compute Tz (Bellman operator T applied to z)
              Tz = replay_data.rewards.to(self.device) + (1 - replay_data.dones.to(self.device)) * self.gamma * self.support
              Tz = Tz.clamp(min=self.v_min, max=self.v_max)
              
              # Compute L2 projection of Tz onto fixed support z
              b = (Tz - self.v_min) / self.delta_z
              l, u = b.floor().long(), b.ceil().long()
              l[(u > 0) * (l == u)] -= 1
              u[(l < (self.n_atoms - 1)) * (l == u)] += 1

              # Distribute probability of Tz
              m = torch.zeros_like(target_distributions)
              offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).unsqueeze(1).expand(batch_size, self.n_atoms).long().to(self.device)
              m.view(-1).index_add_(0, (l + offset).view(-1), (target_distributions * (u.float() - b)).view(-1))
              m.view(-1).index_add_(0, (u + offset).view(-1), (target_distributions * (b - l.float())).view(-1))

          # Get current Q-values estimates
          current_state_distributions = self.q_net(replay_data.observations)
          actions = replay_data.actions.long().flatten().to(self.device)
          current_distribution = current_state_distributions[torch.arange(batch_size, device=self.device), actions]

          # Compute KL divergence
          loss = -(m * torch.log(current_distribution + 1e-8)).sum(dim=1).mean()

          # Optimize the policy
          self.policy.optimizer.zero_grad()
          loss.backward()
          # Clip gradient norm
          torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
          self.policy.optimizer.step()

          # Update target network
          if self._n_calls % self.target_update_interval == 0:
              self.q_net_target.load_state_dict(self.q_net.state_dict())

      # Increase update counter
      self._n_calls += gradient_steps

def create_categorical_dqn(env, **kwargs):
    return CategoricalDQN(CategoricalPolicy, env, **kwargs)