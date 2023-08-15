from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.time_aware_observation import TimeAwareObservation
from .cfg import CFG
import gymnasium

def local_print(local_env: gymnasium.Env):
    print(f"Observation Space: {local_env.observation_space}")
    print(f"Action Space: {local_env.action_space}")
    print(f"Reward Range: {local_env.reward_range}")

def _gray_scale(env: gymnasium.Env) -> gymnasium.Env:
    return GrayScaleObservation(env, keep_dim=True)

def _resize(env: gymnasium.Env) -> gymnasium.Env:
    return ResizeObservation(env, CFG.new_size)

def _time_aware(env: gymnasium.Env) -> gymnasium.Env:
    return TimeAwareObservation(env)

_field = {
    "gray": _gray_scale,
    "resize": _resize,
    "time": _time_aware,
}

class __Env2(gymnasium.Env):
    def __init__(self, env: gymnasium.Env):
        self.env = env

    def __call__(self, param: str) -> gymnasium.Env:
        return _field[param](self.env)

def transform(env: gymnasium.Env):
    return __Env2(env)
