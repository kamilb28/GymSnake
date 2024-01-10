from rl import Qlearn, get_state
from evns.GymSnakeSimpleObsEnv import SnakeSimpleObsEnv
from evns.GymSnakeEnv import SnakeEnv, ACTION_SPACE
from tqdm import tqdm
import pickle
import os
import numpy as np

# ############# import board ############# #
board_file_path = os.path.join(os.path.dirname(__file__),
                               '../../boards/board_001.txt')
with open(board_file_path, 'r') as file:
    board = file.read().splitlines()
# #############              ############# #

env = SnakeSimpleObsEnv(render_mode=None, import_board=board)  # human for rendering
env.metadata["render_fps"] = 100
qlearn = Qlearn()

pickle_in = open('data/240108_1327.pkl', 'rb')  # used data/240108_1327.pkl or data/240108_1325.pkl
qlearn.Q = pickle.load(pickle_in)

terminated = False
truncated = False

obs, _ = env.reset()
state = get_state(obs)

episodic_reward = 0

scores = []

for game in tqdm(range(50)):
    terminated = False
    truncated = False
    obs, _ = env.reset()
    state = get_state(obs)

    episodic_reward = 0

    while not (terminated or truncated):
        action = np.random.choice(list(ACTION_SPACE.values()))
        #action = qlearn.get_max_action(state)

        obs, reward, terminated, truncated, info = env.step(action)
        episodic_reward += reward
        state = get_state(obs)
    scores.append(env.score)

env.close()
print(episodic_reward)
print(np.mean(scores))
print(max(scores))
