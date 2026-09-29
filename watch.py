from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("Pusher-v5", render_mode="human")
model = PPO.load("pusher_model")

obs, info = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()