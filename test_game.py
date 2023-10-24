from evns.GymSnakeEnv import SnakeEnv, ACTION
import numpy as np

if __name__ == '__main__':
    env = SnakeEnv(render_mode="human", size=20)

    terminated = False
    truncated = False

    score = 0

    obs, _ = env.reset()

    while not (terminated or truncated):

        action = np.random.choice([ACTION["TURN_LEFT"], ACTION["DO_NOTHING"], ACTION["TURN_RIGHT"]])
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

    print("Score:", score)

    env.close()
