import gymnasium as gym
import numpy as np
import typing


class En0Wrapper:
    ROW = 210
    COL = 160

    ob0 = np.zeros((ROW, COL), dtype=np.float32)
    rw0 = 0.0

    def __init__(self, name: str) -> None:
        self.name = name
        self.env: gym.Env = self._new_env(name)
        self.A: int = self._action_ranges()
        self.L: int = self._lives()

    def __repr__(self) -> str:
        return f"""
        This is a customized arcade learning environment wrapper.
        This env is created with {self.name},
        whose Action Range is {self.A};
        whose Lives is {self.L}.
        """

    def _lives(self) -> int:
        return self.env.get_wrapper_attr(name="ale").lives()

    def _new_env(self, name: str) -> gym.Env:
        return gym.make(name, obs_type="rgb")

    def _clipped_r(self, reward: typing.SupportsFloat) -> float:
        # Clip the reward to {-1.0, 0.0, 1.0} to care all senarios.
        if reward.__float__ == 0.0:
            return 0.0
        return 1.0 if reward.__float__() > 0 else -1.0

    def _action_ranges(self) -> int:
        # In action space, int64 is not necessary
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return int(self.env.action_space.n)
        raise Exception("action_space should be spaces.Discrete.")

    def act(self) -> int:
        return np.random.randint(self.A)

    def act_from_distrib(self, p: np.ndarray) -> int:
        # action from a behaviour distribution
        # TODO: change to numpy.random.Generator.choice after pyright can understand
        if p.size == self.A:
            return np.random.choice(a=self.A, size=None, replace=False, p=p)
        raise Exception(
            "The given p must contain a probability for each available action"
        )

    def step(self, action: int) -> ...:
        # Treat truncated as a negative reward, while terminated as the real done.
        observation, reward, terminated, truncated, info = self.env.step(action)
        if truncated:
            return observation, -1.0, terminated, info
        else:
            return observation, self._clipped_r(reward), terminated, info

    def reset(self) -> np.ndarray:
        observation, _ = self.env.reset(seed=42)
        return observation
