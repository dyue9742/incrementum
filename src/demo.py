from ale_py import roms
from utils.field import local_print, transform
import gymnasium


gymnasium.logger.set_level(50)
ale_roms = roms.__all__[1:]

env = gymnasium.make(ale_roms[0], render_mode="human")

env = transform(env)("gray")
env = transform(env)("resize")

local_print(env)
observation, info = env.reset(seed=42)

for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()
