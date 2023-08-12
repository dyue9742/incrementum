import gymnasium as gym

import torch
import torch.nn as nn

from ale_py import roms

print(torch.__version__)
ale_roms = roms.__all__[1:]

gym.logger.set_level(80)


def env_print(local_env: gym.Env):
    print(f"Action Space: {local_env.action_space}")
    print(f"Observation Space: {local_env.observation_space}")
    print(f"Reward Range: {local_env.reward_range}")


env = gym.make(ale_roms[0], render_mode="human")
env_print(env)
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
