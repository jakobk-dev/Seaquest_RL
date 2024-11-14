import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import optim
import numpy as np

class NoisyNetwork(nn.Module):
    
    #NoisyNet with factorized gaussian noise
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        n_input_channels = observation_space.shape[0]
        self.n_actions = action_space.n
        
        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),       
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1)   
        )
        
        # compute feature size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            conv_out = self.features(sample_input)
            self.feature_size = conv_out.view(1, -1).size(1)
        
        # Noisy layers
        self.noisy_layers = nn.Sequential(
            NoisyLinear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, self.n_actions)
        )

    def forward(self, x):
        #important normalize inputs      
        x = x.float() / 255.0
        conv_out = self.features(x)
        batch_size = conv_out.shape[0]
        flattened = conv_out.view(batch_size, -1)
        q_values = self.noisy_layers(flattened)
        return q_values

    def reset_noise(self):
        #Reset noise for all noisy layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class NoisyLinear(nn.Module):
    #factorized gaussian noise layer
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
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
        # initialize parameters based on paper 
        mu_range = 1.0 / np.sqrt(self.in_features)
        sigma_range = self.std_init / np.sqrt(self.in_features)
        # initialization for mu weights
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # initialize sigma weights with constant value
        self.weight_sigma.data.fill_(sigma_range)
        self.bias_sigma.data.fill_(sigma_range)

    def _scale_noise(self, size: int) -> torch.Tensor:
        #scale noise for factorized gaussian noise
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        # generate new noise
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #forward pass with noise
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

#policy class for noisy dqn
class NoisyDQNPolicy(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        features_extractor_class=NoisyNetwork,
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
        
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
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
        q_values = self.q_net(obs)
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
#dqn with noisynet exploration
class NoisyDQN(DQN):
    def __init__(
        self,
        policy,
        env,
        learning_rate=1e-4,
        buffer_size=30000,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5000,
        exploration_fraction=0.0,  # not used for noisy
        exploration_initial_eps=0.0,  # not used for noisy
        exploration_final_eps=0.0,  # not used for noisy
        max_grad_norm=10,
        tensorboard_log=None, 
        policy_kwargs=None,
        verbose=1,
        device='auto',
        _init_setup_model=True
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
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

    def train(self, gradient_steps: int, batch_size: int = 128) -> None:
      self.policy.set_training_mode(True)
      
      for _ in range(gradient_steps):
          # Sample replay buffer
          replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

          with torch.no_grad():
              # Reset noise for target network
              self.policy.q_net_target.reset_noise()

              # get the next Q-values using the target network
              next_q_values = self.policy.q_net_target(replay_data.next_observations)
              
              # get highest q value
              next_q_values, _ = next_q_values.max(dim=1)
              # ensure shape is correct
              next_q_values = next_q_values.reshape(-1, 1)
              # td target
              target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

          # Reset the noise for online network
          self.policy.q_net.reset_noise()
          
          # Get current q values estimates
          current_q_values = self.policy.q_net(replay_data.observations)

          # get the q values for the actions from the replay buffer
          current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

          # Huber loss 
          loss = F.smooth_l1_loss(current_q_values, target_q_values)

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

def create_noisy_dqn(env, **kwargs):
    return NoisyDQN(NoisyDQNPolicy, env, **kwargs)