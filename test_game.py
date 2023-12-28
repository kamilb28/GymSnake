from evns.GymSnakeEnv import SnakeEnv, ACTION_SPACE
from evns.GymSnakeSimpleObsEnv import SnakeSimpleObsEnv
import numpy as np

if __name__ == '__main__':

    with open('./boards/board_003.txt', 'r') as file:
        board = file.read().splitlines()

    env = SnakeSimpleObsEnv(render_mode="human", size=5, import_board=board)
    env.metadata["render_fps"] = 10  # for faster rendering

    terminated = False
    truncated = False

    score = 0

    obs, _ = env.reset()

    while not (terminated or truncated):

        action = np.random.choice(list(ACTION_SPACE.values()))
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        score += reward

    print("Score:", score)

    env.close()
