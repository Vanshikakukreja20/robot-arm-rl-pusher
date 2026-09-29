from stable_baselines3 import PPO
import gymnasium as gym
import cv2

env = gym.make("Pusher-v5", render_mode="rgb_array")
model = PPO.load("pusher_model")

frames = []
obs, info = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        obs, info = env.reset()
env.close()

height, width, _ = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('trained_agent.mp4', fourcc, 30, (width, height))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
print("Saved trained_agent.mp4")