import gymnasium as gym
from stable_baselines3 import PPO


# Create the environment
env = gym.make("ALE/Seaquest-v5", render_mode='human')

# Create the RL model
model = PPO("CnnPolicy", env, verbose=1)

# Train the model for 1000 steps
model.learn(total_timesteps=1000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")





