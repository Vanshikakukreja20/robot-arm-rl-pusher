import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.monitor.csv", skiprows=1)
df['episode'] = range(1, len(df) + 1)

plt.figure(figsize=(10, 5))
plt.plot(df['episode'], df['r'], alpha=0.3, label='Episode reward')
plt.plot(df['episode'], df['r'].rolling(50).mean(), label='50-episode rolling average', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress: Pusher-v5 PPO')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_curve.png', dpi=150)
print("Saved training_curve.png")
plt.show()