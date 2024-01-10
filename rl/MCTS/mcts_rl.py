from evns.GymSnakeEnv import SnakeEnv, ACTION_SPACE
import math
import random
import time
import os
import numpy as np

class Node:
    def __init__(self, state, action, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.rewards = 0
        self.visits = 0
        self.actions_to_try = list(ACTION_SPACE.values())

    def is_terminal(self):
        return self.state["terminated"] or self.state["truncated"]

    def get_next_state(self, action):
        env = SnakeEnv()
        env.game_state_insert(self.state)
        _ = env.step(action)
        return env.game_state_copy()

    def expand(self):
        action = self.actions_to_try.pop()
        next_state = self.get_next_state(action)
        child_node = Node(next_state, action, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.rewards += self.state["reward"] + reward

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.rewards / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]


def simulate_is_terminal(state):
    return state["terminated"] or state["truncated"]


def simulate_next_state(state, action):
    env = SnakeEnv()
    env.game_state_insert(state)
    _ = env.step(action)
    return env.game_state_copy()


class MCTS:
    def __init__(self, time_limit=None):
        self.time_limit = time_limit
        self.start_time = time.time()
        self.root = None

    def choose(self, state):
        self.root = Node(state, None)

        while not self.time_limit_reached():
            node = self.select_node(self.root)
            result = self.simulate(node)
            self.backpropagate(node, result)

        return self.root.best_child().action

    def select_node(self, node):
        while not node.is_terminal():
            if node.actions_to_try:
                return node.expand()
            else:
                node = node.best_child()
        return node

    def simulate(self, node):
        current_state = node.state
        while not simulate_is_terminal(current_state):
            action = random.choice(list(ACTION_SPACE.values()))
            current_state = simulate_next_state(current_state, action)
        return current_state["reward"]

    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def time_limit_reached(self):
        return (time.time() - self.start_time)*1000 > self.time_limit


# ############# import board ############# #
board_file_path = os.path.join(os.path.dirname(__file__),
                               '../../boards/board_000.txt')
with open(board_file_path, 'r') as file:
    board = file.read().splitlines()
# #############              ############# #

env = SnakeEnv(render_mode="human", import_board=board)
terminated = False
truncated = False

_ = env.reset()
while not (terminated or truncated):
    action = np.random.choice(list(ACTION_SPACE.values()))
    #action = qlearn.get_max_action(state)
    mcts = MCTS(250)
    act = mcts.choose(env.game_state_copy())

    _, _, terminated, truncated, _ = env.step(act)

env.close()
