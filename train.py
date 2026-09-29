from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

env = gym.make("Pusher-v5")
env = Monitor(env, filename="training_log")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=200_000)

model.save("pusher_model")
print("Training done, model saved.")