import gymnasium as gym
from gym import spaces
import pygame
import numpy as np

DIRECTION = {
    "RIGHT": 0,
    "UP": 1,
    "LEFT": 2,
    "DOWN": 3
}

ACTION = {
    "TURN_LEFT": 0,
    "DO_NOTHING": 1,
    "TURN_RIGHT": 2
}

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 40}
    num_of_steps = 0
    terminated = False
    truncated = False

    def __init__(self, render_mode=None, size=10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.board = 

        self.observation_space = spaces.Space(
            {
                "head": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "body": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "direction": spaces.Box(0, 2, shape=(1,), dtype=int),
                "fruit": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Space([ACTION["TURN_LEFT"], ACTION["DO_NOTHING"], ACTION["TURN_RIGHT"]])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"head": self._head_location,
                "body": self._body_location,
                "direction": self._head_direction,
                "fruit": self._fruit_location}

    def _get_info(self):
        # manhattan distance between snake head and fruit
        return {"distance": np.linalg.norm(self._head_location - self._fruit_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._reset_data()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def _reset_data(self):
        self._reset_snake_position()
        self._reset_fruit_position()

        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.num_of_steps = 0

    def _reset_snake_position(self):
        self._head_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._body_location = []
        self._body_location.append(self._head_location)
        self._head_direction = DIRECTION["RIGHT"]

    def _reset_fruit_position(self):
        # sample the target's location randomly until it does not coincide with the snake's body
        self._fruit_location = self._head_location
        while any(np.array_equal(self._fruit_location, snake_body) for snake_body in self._body_location):
            self._fruit_location = self.np_random.integers(0, self.size, size=2, dtype=int)

    def step(self, action):
        self.num_of_steps += 1
        self._update_snake_location(action)

        if np.array_equal(self._head_location, self._fruit_location):
            self.reward += 10  # update reward
            self.num_of_steps = 0
            self._reset_fruit_position()

        # snake going in circles
        self.truncated = True if self.num_of_steps >= (self.size**2 * 2) else False

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self.reward, self.terminated, self.truncated, self._get_info()

    def _update_snake_location(self, action):
        self._head_direction = self.change_direction_based_on_action(self._head_direction, action)

        self._head_location = self.calculate_next_location(
            from_location=self._head_location,
            direction=self._head_direction
        )

        # update the body (follow head)
        self._body_location.insert(0, self._head_location)
        if not np.array_equal(self._head_location, self._fruit_location):
            self._body_location.pop()

        self.terminated = self.check_if_dead()

    def check_if_dead(self) -> bool:
        if any(np.array_equal(self._head_location, body) for body in self._body_location[1:]):
            self.reward -= 100
            return True
        if self.position_out_of_borders(self._head_location):
            self.reward -= 100
            return True
        return False

    def position_out_of_borders(self, position) -> bool:
        return (position[0] < 0) \
            or (position[0] > self.size - 1) \
            or (position[1] < 0)\
            or (position[1] > self.size - 1)

    def change_direction_based_on_action(self, direction, action) -> int:
        if action != ACTION["DO_NOTHING"]:
            if action == ACTION["TURN_LEFT"]:
                direction -= 1
            elif action == ACTION["TURN_RIGHT"]:
                direction += 1

        if direction > 3:
            direction = 0
        if direction < 0:
            direction = 3

        return direction

    def calculate_next_location(self, from_location, direction) -> np.array:
        direction_to_arr = {
            DIRECTION["RIGHT"]: np.array([1, 0]),
            DIRECTION["UP"]: np.array([0, 1]),
            DIRECTION["LEFT"]: np.array([-1, 0]),
            DIRECTION["DOWN"]: np.array([0, -1]),
        }
        direction_arr = direction_to_arr[direction]

        next_location = from_location + direction_arr

        # 'teleport' to next side of board
        if next_location[0] < 0:
            next_location[0] = self.size - 1
        elif next_location[0] > self.size - 1:
            next_location[0] = 0

        if next_location[1] < 0:
            next_location[1] = self.size - 1
        elif next_location[1] > self.size - 1:
            next_location[1] = 0

        return next_location

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # draw the body
        for body in self._body_location:
            pygame.draw.rect(
                canvas,
                (255, 183, 0),
                pygame.Rect(
                    pix_square_size * body,
                    (pix_square_size, pix_square_size),
                ),
            )

        # draw the head
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._head_location,
                (pix_square_size, pix_square_size),
            ),
        )
        #draw the fruit
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._fruit_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (222, 235, 255),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (222, 235, 255),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            # Check for window close event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            # delay, to keep stable framerate
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

        self.terminated = True
