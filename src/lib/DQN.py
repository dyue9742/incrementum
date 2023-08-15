from typing import DefaultDict
import numpy as np
import gymnasium

class Q:
    def __init__(
        self,
        env: gymnasium.Env,
        learning_rate: float,
        epsilon_begin: float,
        epsilon_decay: float,
        epsilon_final: float,
        discount_factor: float = 0.95,
        ):
        self.q_values = DefaultDict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.dc = discount_factor
        self.e1 = epsilon_begin
        self.ex = epsilon_decay
        self.e2 = epsilon_final

        self.training_error = []

