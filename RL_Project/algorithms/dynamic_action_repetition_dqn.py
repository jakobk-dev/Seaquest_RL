import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import BasePolicy
from torch import optim
import gym
import numpy as np

# Currently, this algorithm was not run or implmented with a graph in the paper.
class ActionRepetitionMapper:
    """
    Maps action indices to (action, repetition) pairs and vice versa.
    """
    def __init__(self, num_actions, repetition_factors):
        self.num_actions = num_actions
        self.repetition_factors = repetition_factors
        self.action_repetition_pairs = [
            (action, repetition)
            for action in range(num_actions)
            for repetition in repetition_factors
        ]
        self.total_actions = len(self.action_repetition_pairs)

    def get_action_repetition(self, action_index):
        return self.action_repetition_pairs[action_index]

    def get_action_index(self, action, repetition):
        return self.action_repetition_pairs.index((action, repetition))


class ActionRepetitionEnvWrapper(gym.Wrapper):
    """
    Environment wrapper that handles action repetitions.
    """
    def __init__(self, env, repetition_factors):
        super().__init__(env)
        self.repetition_factors = repetition_factors
        self.num_actions = env.action_space.n
        self.mapper = ActionRepetitionMapper(self.num_actions, repetition_factors)
        # Adjust the action space to include action repetitions
        self.action_space = gym.spaces.Discrete(self.mapper.total_actions)

    def step(self, action_index):
        action, repetition = self.mapper.get_action_repetition(action_index)
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(repetition):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class DynamicActionRepetitionDQN(DQN):
    """
    Dynamic Action Repetition DQN agent that extends DQN to handle action repetitions.
    """
    def __init__(
        self,
        policy,
        env,
        repetition_factors=[1, 2, 4],
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
        # Wrap the environment
        env = ActionRepetitionEnvWrapper(env, repetition_factors)
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
        self.repetition_factors = repetition_factors
        self.num_actions = env.unwrapped.action_space.n
        self.mapper = env.mapper

    # Optionally, you can override methods or add additional methods if necessary


def create_dynamic_action_repetition_dqn(env, **kwargs):
    """
    Factory function to create a Dynamic Action Repetition DQN agent.
    """
    return DynamicActionRepetitionDQN("MlpPolicy", env, **kwargs)
