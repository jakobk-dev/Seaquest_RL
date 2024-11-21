import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from torch import optim
import numpy as np

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.1):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorized noise parameters
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        sigma_range = self.std_init / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_range)
        self.bias_sigma.data.fill_(sigma_range)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class NoisyCategoricalNetwork(nn.Module):
    def __init__(self, observation_space, action_space, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        
        n_input_channels = observation_space.shape[0]
        self.n_actions = action_space.n
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)
        
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1)
        )
        
        # Compute feature size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            conv_out = self.features(sample_input)
            self.feature_size = conv_out.view(1, -1).size(1)
        
        # Noisy fully connected layers
        self.noisy_layers = nn.Sequential(
            NoisyLinear(self.feature_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, self.n_actions * self.n_atoms)
        )

    def forward(self, x):
        x = x.float() / 255.0
        batch_size = x.shape[0]
        conv_out = self.features(x)
        flattened = conv_out.view(batch_size, -1)
        out = self.noisy_layers(flattened)
        out = out.view(batch_size, self.n_actions, self.n_atoms)
        return F.softmax(out, dim=2)
    
    def get_q_values(self, x):
        distribution = self.forward(x)
        support = self.support.to(distribution.device)
        q_values = (distribution * support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        return q_values

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
class NoisyCategoricalPolicy(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_atoms=51,
        v_min=-10,
        v_max=10,
        net_arch=None,
        activation_fn=nn.ReLU,
        features_extractor_class=NoisyCategoricalNetwork,
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
        self.v_min = v_min
        self.v_max = v_max
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_atoms": self.n_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max
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

class NoisyCategoricalDQN(DQN):
    def __init__(
        self,
        policy,
        env,
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        learning_rate=1e-4,
        buffer_size=30000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.0,  # Not used with noisy nets
        exploration_initial_eps=0.0,  # Not used with noisy nets
        exploration_final_eps=0.0,  # Not used with noisy nets
        max_grad_norm=10,
        tensorboard_log = None,
        policy_kwargs=None,
        verbose=1,
        device='auto',
        _init_setup_model=True
    ):
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.update({
            "n_atoms": n_atoms,
            "v_min": v_min,
            "v_max": v_max,
            "optimizer_class": optim.Adam,
            "optimizer_kwargs": {"eps": 0.01/batch_size}
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
            _init_setup_model=_init_setup_model
        )
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

    def train(self, gradient_steps: int, batch_size: int = 32) -> None:
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            # sample the replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # reset noise for target network
                self.policy.q_net_target.reset_noise()
                
                # Distributional Bellman Update
                # get next state distributions from target network
                next_state_distributions = self.policy.q_net_target(replay_data.next_observations)
                
                # select best action 
                next_q_values = (next_state_distributions * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
                next_actions = next_q_values.argmax(dim=1).reshape(-1, 1)
                
                # get the target distribution
                target_distributions = next_state_distributions[torch.arange(batch_size, device=self.device), next_actions.squeeze()]
                
                # apply bellman adding rewards and support
                Tz = replay_data.rewards.to(self.device) + (1 - replay_data.dones.to(self.device)) * self.gamma * self.support
                Tz = Tz.clamp(min=self.v_min, max=self.v_max)
                
                # calculate where new values fall between the atoms
                b = (Tz - self.v_min) / self.delta_z
                # get lower and upper bound of atoms
                l, u = b.floor().long(), b.ceil().long()
                # edge case where value is exactly on atom
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.n_atoms - 1)) * (l == u)] += 1

                # Distribution update
                # initialize target distribution
                m = torch.zeros_like(target_distributions)
                # offset for batch for correctly distributing probabilities to current atom in the flatten tensor
                offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).unsqueeze(1).expand(batch_size, self.n_atoms).long().to(self.device)
                # distribute probabilities for lower atoms
                m.view(-1).index_add_(0, (l + offset).view(-1), (target_distributions * (u.float() - b)).view(-1))
                # and upper atoms
                m.view(-1).index_add_(0, (u + offset).view(-1), (target_distributions * (b - l.float())).view(-1))

            # reset noise for online network
            self.policy.q_net.reset_noise()
            
            # cross entroy loss update
            # get current distribution
            current_state_distributions = self.policy.q_net(replay_data.observations)
            actions = replay_data.actions.long().flatten()
            current_distribution = current_state_distributions[torch.arange(batch_size, device=self.device), actions]

            # compute cross entropy loss between current distribution and target distribtion m 
            loss = -(m * torch.log(current_distribution + 1e-8)).sum(dim=1).mean()

            # optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # update target network
            if self._n_calls % self.target_update_interval == 0:
                self.policy.q_net_target.load_state_dict(self.policy.q_net.state_dict())
        # increase update counter
        self._n_calls += gradient_steps

def create_noisy_categorical_dqn(env, **kwargs):
    return NoisyCategoricalDQN(NoisyCategoricalPolicy, env, **kwargs)