import gymnasium as gym
from gym import spaces
import pygame
import numpy as np
import random

DIRECTION = {
    "RIGHT": 0,
    "UP": 1,
    "LEFT": 2,
    "DOWN": 3
}

DIRECTION_WSAD = {
    "W": DIRECTION["UP"],
    "S": DIRECTION["DOWN"],
    "A": DIRECTION["LEFT"],
    "D": DIRECTION["RIGHT"],
    "R": None  # for random
}

ACTION_SPACE = {
    "TURN_LEFT": 0,
    "DO_NOTHING": 1,
    "TURN_RIGHT": 2
}


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    size = 10  # The size of the square grid
    window_size = 512  # The size of the PyGame window
    walls = []  # cords of the wall
    start_position = {}
    num_of_steps = 0
    terminated = False
    truncated = False
    death_reward = -100
    truncated_reward = -100
    fruit_reward = 100
    closing_to_reward = 1
    away_from_reward = -1

    def __init__(self, render_mode=None, size=10, import_board=None):
        if import_board is None:
            self.size = size  # The size of the square grid
        else:
            self.import_board(import_board)

        self.observation_space = spaces.Space(
            {
                "head": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "body": spaces.Box(0, size - 1, shape=(size, 2), dtype=int),
                "direction": spaces.Discrete(4),
                "fruit": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "walls": spaces.Box(0, size - 1, shape=(len(self.walls), 2), dtype=int)
            }
        )
        self.action_space = spaces.Space([ACTION_SPACE["TURN_LEFT"], ACTION_SPACE["DO_NOTHING"], ACTION_SPACE["TURN_RIGHT"]])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.reward = 0
        self.window = None
        self.clock = None

    def import_board(self, board):
        self.size = len(board[0])

        for i, row in enumerate(board):
            for j, char in enumerate(row):
                if char == '#':
                    self.walls.append(np.array([i, j]))
                if char in ('W', 'S', 'A', 'D', 'R'):  # R is for random
                    self.start_position["head"] = np.array([i, j])
                    self.start_position["direction"] = DIRECTION_WSAD[char]

    def _get_obs(self):
        return {"head": self._head_location,
                "body": self._body_location,
                "direction": self._head_direction,
                "fruit": self._fruit_location,
                "walls": self.walls}

    def _get_info(self):
        # manhattan distance between snake head and fruit
        return {"distance": np.linalg.norm(self._head_location - self._fruit_location, ord=1)}

    def reset(self, seed=None, options=None):
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
        self.score = 0
        self.num_of_steps = 0

    def _reset_snake_position(self):
        self._head_location = self.start_position.get("head", self.np_random.integers(0, self.size, size=2, dtype=int))
        self._head_direction = self.start_position.get("direction", self.np_random.choice(list(DIRECTION.values())))
        while any(np.array_equal(self._head_location, wall) for wall in self.walls):
            self._head_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._body_location = []
        self._body_location.append(self._head_location)

    def _reset_fruit_position(self):
        # # sample the target's location randomly until it does not coincide with the snake's body  ### INEFFICIENT!!!
        # self._fruit_location = self._head_location
        # while any(np.array_equal(self._fruit_location, snake_body) for snake_body in self._body_location):
        #     while any(np.array_equal(self._fruit_location, wall) for wall in self.walls):
        #         self._fruit_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # Create a list of valid fruit positions
        valid_positions = []
        for x in range(self.size):
            for y in range(self.size):
                position = np.array([x, y])
                if not any(np.array_equal(position, snake_body) for snake_body in self._body_location) and \
                        not any(np.array_equal(position, wall) for wall in self.walls):
                    valid_positions.append(position)

        if len(valid_positions) < 1:
            self.truncated = True
            self.terminated = True
            self.reward += 100
            return

        # Choose a random valid position for the fruit
        self._fruit_location = random.choice(valid_positions)

    def step(self, action):
        self.reward = 0
        self.num_of_steps += 1
        self.reward += self._update_snake_location(action)

        if not self.terminated and np.array_equal(self._head_location, self._fruit_location):
            self.reward += self.fruit_reward  # update reward
            self.score += 1
            self.num_of_steps = 0
            self._reset_fruit_position()

        # snake going in circles
        # self.truncated = True if self.num_of_steps >= (self.size**2 * 2) else False
        if not self.terminated and self.num_of_steps >= (self.size**2):
            self.truncated = True
            self.reward += self.truncated_reward

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self.reward, self.terminated, self.truncated, self._get_info()

    def _update_snake_location(self, action):
        old_distance_to_fruit = np.linalg.norm(self._head_location - self._fruit_location, ord=1)

        self._head_direction = self.change_direction_based_on_action(self._head_direction, action)

        self._head_location = self.calculate_next_location(
            from_location=self._head_location,
            direction=self._head_direction
        )

        # update the body (follow head)
        self._body_location.insert(0, self._head_location)
        if not np.array_equal(self._head_location, self._fruit_location):
            self._body_location.pop()

        self.terminated, reward_from_death = self.check_if_dead()

        new_distance_to_fruit = np.linalg.norm(self._head_location - self._fruit_location, ord=1)
        reward_from_distance = self.closing_to_reward if new_distance_to_fruit < old_distance_to_fruit \
            else self.away_from_reward

        return reward_from_death + reward_from_distance

    def check_if_dead(self) -> (bool, int):
        if any(np.array_equal(self._head_location, body) for body in self._body_location[1:]):
            return True, self.death_reward
        if any(np.array_equal(self._head_location, wall) for wall in self.walls):
            return True, self.death_reward
        return False, 0

    def change_direction_based_on_action(self, direction, action) -> int:
        if action != ACTION_SPACE["DO_NOTHING"]:
            if action == ACTION_SPACE["TURN_LEFT"]:
                direction -= 1
            elif action == ACTION_SPACE["TURN_RIGHT"]:
                direction += 1

        if direction > 3:
            direction = 0
        if direction < 0:
            direction = 3

        return direction

    def calculate_next_location(self, from_location, direction) -> np.array:
        direction_to_arr = {
            DIRECTION["RIGHT"]: np.array([1, 0]),
            DIRECTION["UP"]: np.array([0, -1]),
            DIRECTION["LEFT"]: np.array([-1, 0]),
            DIRECTION["DOWN"]: np.array([0, 1]),
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

        # draw walls
        for wall in self.walls:
            pygame.draw.rect(
                canvas,
                (0, 0, 255),
                pygame.Rect(
                    pix_square_size * wall,
                    (pix_square_size, pix_square_size),
                ),
            )

        # grid lines
        for x in range(self.size + 1):
            # horizontal lines
            pygame.draw.line(
                canvas,
                (222, 235, 255),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            # vertical lines
            pygame.draw.line(
                canvas,
                (222, 235, 255),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
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

    def game_state_copy(self):
        return {
            "size": self.size,
            "_head_location": self._head_location.copy(),
            "_body_location": self._body_location.copy(),
            "_fruit_location": self._fruit_location.copy(),
            "_head_direction": self._head_direction,
            "walls": self.walls.copy(),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reward": self.reward,
            "score": self.score,
            "num_of_steps": self.num_of_steps
        }

    def game_state_insert(self, params: dict):
        self.size = params["size"]
        self._head_location = params["_head_location"]
        self._body_location = params["_body_location"]
        self._fruit_location = params["_fruit_location"]
        self._head_direction = params["_head_direction"]
        self.walls = params["walls"]
        self.terminated = params["terminated"]
        self.truncated = params["truncated"]
        self.reward = params["reward"]
        self.score = params["score"]
        self.num_of_steps = params["num_of_steps"]
