import numpy as np
import pandas as pd
from tqdm import tqdm
from evns.GymSnakeSimpleObsEnv import SnakeSimpleObsEnv
from evns.GymSnakeEnv import ACTION_SPACE
import pickle
import matplotlib.pyplot as plt
import datetime
import os

class Qlearn:
    Q = {}

    def __init__(self, alpha=0.01, gamma=0.99):
        self._alpha = alpha
        self._gamma = gamma
        self.fill_Q()

    # def get_all_possible_states(self):
    #     import itertools
    #
    #     # all possible states for the observation space
    #     danger_states = list(itertools.product([0, 1], repeat=3))
    #     direction_states = list(itertools.product([0, 1], repeat=4))
    #     food_sense_states = list(itertools.product([0, 1], repeat=4))
    #
    #     # all possible combinations of states
    #     all_states = []
    #     for danger in danger_states:
    #         for direction in direction_states:
    #             for food_sense in food_sense_states:
    #                 all_states.append(list(np.concatenate(danger, direction, food_sense)))
    #
    #     return all_states

    def get_all_possible_states(self):  # faster!
        import itertools
        array_size = 11  # example state: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        possible_values = [0, 1]
        return np.array(list(itertools.product(possible_values, repeat=array_size)))

    def fill_Q(self):
        for state in self.get_all_possible_states():
            for action in list(ACTION_SPACE.values()):
                self.Q[tuple(state), action] = 0

    def update_Q(self, state, action, state_new, action_new):
        self.Q[state, action] = self.Q[state, action] + \
                                self._alpha*(reward + \
                                self._gamma * self.Q[state_new, action_new] - \
                                self.Q[state, action])

    def get_max_action(self, state):
        actions_for_state = np.array([self.Q[tuple(state), action] for action in ACTION_SPACE.values()])
        return np.argmax(actions_for_state)


def get_state(observation):
    return tuple(np.concatenate([
        observation["danger"].astype('int32'),
        observation["direction"].astype('int32'),
        observation["food_sense"].astype('int32')
    ]))


# ############# import board ############# #
board_file_path = os.path.join(os.path.dirname(__file__),
                               '../../boards/board_000.txt')
with open(board_file_path, 'r') as file:
    board = file.read().splitlines()
# #############              ############# #

if __name__ == '__main__':
    num_of_episodes = 20000
    eps = 1.  # exploration parameter

    episode_rewards = []
    episode_epsilons = []

    env = SnakeSimpleObsEnv(render_mode=None, import_board=board)
    env.metadata["render_fps"] = 50  # for faster rendering
    qlearn = Qlearn(alpha=0.1, gamma=0.99)

    # learning loop
    progress_bar = tqdm(total=num_of_episodes, desc="Learning")
    for episode in range(num_of_episodes):
        terminated = False
        truncated = False

        obs, _ = env.reset()
        state = get_state(obs)

        episodic_reward = 0
        while not (terminated or truncated):
            if np.random.random() < eps:
                action = np.random.choice(list(ACTION_SPACE.values()))
            else:
                action = qlearn.get_max_action(state)

            obs, reward, terminated, truncated, info = env.step(action)
            episodic_reward += reward

            state_new = get_state(obs)
            action_new = qlearn.get_max_action(state)
            qlearn.update_Q(state, action, state_new, action_new)

            state = state_new

        episode_rewards.append(episodic_reward)
        episode_epsilons.append(eps)

        # Update progress bar
        progress_bar.set_postfix(epsilon=f'{eps:.2f}',
                                 reward=f'{episodic_reward}')
        progress_bar.update(1)

        # decrease epsilon over time (in halfway selection strategy will be almost entirely greedy)
        eps = eps - (2 / num_of_episodes) if eps > 0.01 else 0.01

    env.close()

    # save Q
    date = datetime.datetime.now().strftime("%y%m%d_%H%M")
    f = open(f"data/{date}.pkl", "wb")
    pickle.dump(qlearn.Q, f)
    f.close()

    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Episodic Return')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    ax2 = plt.gca().twinx()
    ax2.plot(episode_epsilons, color='red')
    ax2.set_ylabel('Epsilon')
    plt.grid(True)
    plt.savefig(f"data/{date}.png")
    plt.show()

    pd.DataFrame({
        "Episodes": np.array(),
        "Episodic Return": episode_rewards,
        "Epsilon": episode_epsilons
    }).to_csv(f"data/{date}.csv")
