# Robot Arm Pick-and-Push with Reinforcement Learning

## What this is
Trained a 7-joint robotic arm (MuJoCo Pusher-v5 environment) to push an object 
to a target location using Proximal Policy Optimization (PPO) via Stable-Baselines3.

## Setup
- Environment: Gymnasium Pusher-v5 (MuJoCo physics)
- Algorithm: PPO (Stable-Baselines3)
- Training: 200,000 timesteps, ~3.5 minutes on CPU

## Results
Episode reward improved from approximately -47.5 to -38.6 over training 
(see training_curve.png). The trained policy shows visibly more purposeful 
arm movement toward the target compared to a random baseline.

## What I'd try next
- Longer training (500k-1M timesteps) to see if reward continues improving
- Reward shaping: penalize jerky joint movements for smoother motion
- Domain randomization (object mass/friction) to test robustness

## Files
- `train.py` — training script
- `watch.py` — visualize trained policy
- `record.py` — save a video of the trained policy
- `plot_results.py` — generate training curve
- `trained_agent.mp4` — video of the trained arm
- `training_curve.png` — reward over training