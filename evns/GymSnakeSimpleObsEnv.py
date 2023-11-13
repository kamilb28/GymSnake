import gymnasium as gym
from gym import spaces
from evns.GymSnakeEnv import SnakeEnv
import numpy as np

class SnakeSimpleObsEnv(SnakeEnv):
    def __init__(self, render_mode=None, size=10, import_board=None):
        super().__init__(render_mode, size, import_board)

        self.observation_space = spaces.Space(
            {
                "danger": spaces.Box(0, 1, shape=(3,), dtype=int),  # danger: [on left, center, on right]
                "direction": spaces.Box(0, 3, shape=(1,), dtype=int), # direction [int]
                "food": spaces.Box(0, 1, shape=(2,), dtype=int)  # food: [up/down, right/left]
            }
        )

    def _get_obs(self):
        return {
            "danger": None, # TODO
            "direction": self._head_direction,
            "food": self._sense_food()
        }

    def _sense_food(self):
        return np.array([
            self._fruit_location[0] < self._head_location[0],  # food is in left
            self._fruit_location[0] > self._head_location[0],  # food is in right
            self._fruit_location[1] < self._head_location[1],  # food is up
            self._fruit_location[1] > self._head_location[1]   # food is down
        ])