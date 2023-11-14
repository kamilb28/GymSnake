import gymnasium as gym
from gym import spaces
from evns.GymSnakeEnv import SnakeEnv
import numpy as np

class SnakeSimpleObsEnv(SnakeEnv):
    def __init__(self, render_mode=None, size=10, import_board=None):
        super().__init__(render_mode, size, import_board)

        self.observation_space = spaces.Space(
            {
                "danger": spaces.Box(0, 1, shape=(3,), dtype=bool),  # danger: [on left, center, on right]
                "direction": spaces.Box(0, 1, shape=(4,), dtype=bool),  # direction: [up, left, down, right]
                "food_sense": spaces.Box(0, 1, shape=(4,), dtype=bool)  # food: [up, down, right, left]
            }
        )

    def _get_obs(self):
        return {
            "danger": self._danger_array(),
            "direction": self._direction_array(),
            "food_sense": self._sense_food()
        }

    def _positions_around_head(self):
        # position on the left
        direction_to_left = self.change_direction_based_on_action(self._head_direction, 0)  # 0 - go left
        position_left_from_head = self.calculate_next_location(
            self._head_location, direction_to_left
        )
        # position ahead
        position_ahead = self.calculate_next_location(
            self._head_location, self._head_direction
        )
        # position on the right
        direction_to_right = self.change_direction_based_on_action(self._head_direction, 2)  # 2 - go right
        position_right_from_head = self.calculate_next_location(
            self._head_location, direction_to_right
        )

        return [position_left_from_head, position_ahead, position_right_from_head]

    def _danger_array(self):
        return np.array([
            (
                    any(np.array_equal(position, body) for body in self._body_location) or
                    any(np.array_equal(position, body) for body in self.walls)
            ) for position in self._positions_around_head()
        ])

    def _direction_array(self):
        direction_arr = np.array([False, False, False, False])
        direction_arr[self._head_direction] = True
        return direction_arr

    def _sense_food(self):
        return np.array([
            self._fruit_location[0] < self._head_location[0],  # food is in left
            self._fruit_location[0] > self._head_location[0],  # food is in right
            self._fruit_location[1] < self._head_location[1],  # food is up
            self._fruit_location[1] > self._head_location[1]   # food is down
        ])