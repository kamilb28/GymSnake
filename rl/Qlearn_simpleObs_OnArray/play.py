from rl import Qlearn, get_state
from evns.GymSnakeSimpleObsEnv import SnakeSimpleObsEnv
import pickle
import os

# ############# import board ############# #
board_file_path = os.path.join(os.path.dirname(__file__),
                               '../../boards/board_002.txt')
with open(board_file_path, 'r') as file:
    board = file.read().splitlines()
# #############              ############# #

env = SnakeSimpleObsEnv(render_mode="human", import_board=board)
env.metadata["render_fps"] = 20
qlearn = Qlearn()

pickle_in = open('data/231229_1235.pkl', 'rb')
qlearn.Q = pickle.load(pickle_in)

terminated = False
truncated = False

obs, _ = env.reset()
state = get_state(obs)

episodic_reward = 0

while not (terminated or truncated):
    action = qlearn.get_max_action(state)

    obs, reward, terminated, truncated, info = env.step(action)
    episodic_reward += reward
    state = get_state(obs)

env.close()
print(episodic_reward)
