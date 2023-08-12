import gymnasium
from gymnasium.core import ObservationWrapper, RewardWrapper, ActionWrapper
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.resize_observation import ResizeObservation


class Perception(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.env = env
