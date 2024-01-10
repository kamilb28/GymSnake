from rl.Qlearn.rl import get_state
import dqn_rl
from dqn_rl import DQN
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
dqn = DQN(gamma=.95, batch_size=50, learning_rate=0.00025)

dqn.load_model('data/240109_2136.keras')

terminated = False
truncated = False

obs, _ = env.reset()
state = get_state(obs)

scores = []
for game in tqdm(range(50)):
    terminated = False
    truncated = False
    obs, _ = env.reset()
    state = get_state(obs)
    state = np.reshape(state, (1, len(state)))

    episodic_reward = 0

    while not (terminated or truncated):
        #action = np.random.choice(list(ACTION_SPACE.values()))
        action = dqn.get_action(state, epsilon=0)

        obs, reward, terminated, truncated, info = env.step(action)
        state = get_state(obs)
        state = np.reshape(state, (1, len(state)))
    scores.append(env.score)

env.close()
print(np.mean(scores))
print(max(scores))