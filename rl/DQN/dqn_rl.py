import random
import os

import keras.src.callbacks
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import datetime
import pandas as pd
from evns.GymSnakeEnv import ACTION_SPACE
from evns.GymSnakeSimpleObsEnv import SnakeSimpleObsEnv
from rl.Qlearn.rl import get_state

len_state_space = 11


class DQN:

    def __init__(self, gamma, batch_size, learning_rate):
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(len_state_space,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(ACTION_SPACE), activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, state_new, terminated, truncated):
        self.memory.append((state, action, reward, state_new, terminated, truncated))

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(list(ACTION_SPACE.values()))
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        terminateds = np.array([i[4] for i in minibatch])
        truncateds = np.array([i[5] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - terminateds)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)

    def save_model(self, filename=None):
        if filename is None: return
        self.model.save(filename + ".keras")

    def load_model(self, filename):
        if filename is None or filename.split(".")[-1] != "keras": return
        self.model = keras.models.load_model(filename)

# ############# import board ############# #
board_file_path = os.path.join(os.path.dirname(__file__),
                               '../../boards/board_000.txt')
with open(board_file_path, 'r') as file:
    board = file.read().splitlines()
# #############              ############# #


if __name__ == '__main__':
    num_of_episodes = 200
    eps, initial_eps = 1., 1.  # exploration parameters
    decay_rate = -np.log(0.05) / ((num_of_episodes // 6) * 5)

    epsilon_min = .01
    epsilon_decay = .995

    episode_rewards = []
    episode_scores = []
    episode_epsilons = []

    env = SnakeSimpleObsEnv(render_mode=None, size=10, import_board=board)
    env.metadata["render_fps"] = 50
    dqn = DQN(gamma=.95, batch_size=1, learning_rate=0.00025)

    progress_bar = tqdm(total=num_of_episodes, desc="Learning")
    for episode in range(num_of_episodes):
        terminated = False
        truncated = False

        obs, _ = env.reset()
        state = get_state(obs)
        state = np.reshape(state, (1, len(state)))

        episodic_reward = 0
        while not (terminated or truncated):
            action = dqn.get_action(state, epsilon=eps)

            obs, reward, terminated, truncated, info = env.step(action)
            episodic_reward += reward

            state_new = get_state(obs)
            state_new = np.reshape(state_new, (1, len(state_new)))
            dqn.remember(state, action, reward, state_new, int(terminated), int(truncated))
            state = state_new
            if 1 < dqn.batch_size <= len(dqn.memory):
                dqn.replay()
            eps = initial_eps * np.exp(-decay_rate * episode)

        episode_rewards.append(episodic_reward)
        episode_scores.append(env.score)
        episode_epsilons.append(eps)

        # Update progress bar
        progress_bar.set_postfix(epsilon=f'{eps:.2f}',
                                 reward=f'{episodic_reward}')
        progress_bar.update(1)

        # eps = initial_eps * np.exp(-decay_rate * episode)
        # eps = eps - (1.5 / num_of_episodes) if eps > 0.01 else 0.01

    env.close()

    ##############################################################################//////////////

    date = datetime.datetime.now().strftime("%y%m%d_%H%M")

    dqn.save_model(f"data/{date}")

    # Calculate mean rewards for every 100 episodes
    mean_step = num_of_episodes // 10 if num_of_episodes // 10 > 1 else 1
    mean_rewards = [sum(episode_rewards[i:i + mean_step]) / mean_step for i in
                    range(0, len(episode_rewards), mean_step)]

    plt.figure(figsize=(10, 6))

    # Plot on the primary y-axis
    reward_line, = plt.plot(episode_rewards, label='Reward')
    mean_reward_line, = plt.plot(range(0, len(episode_rewards), mean_step), mean_rewards, label='Mean Reward',
                                 linestyle='--')
    plt.title('Episodic Return')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)

    # Plot on the secondary y-axis
    ax2 = plt.gca().twinx()
    epsilon_line, = ax2.plot(episode_epsilons, color='red', label='Epsilon')
    ax2.set_ylabel('Epsilon')

    # Create a single legend for all lines
    plt.legend(handles=[reward_line, mean_reward_line, epsilon_line], labels=['Reward', 'Mean Reward', 'Epsilon'],
               loc='upper right')

    plt.savefig(f"data/{date}.png")
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.plot(episode_scores)
    plt.title('Scores')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(f"data/scores-{date}.png")
    plt.show()

    pd.DataFrame({
        "Episodes": np.arange(1, num_of_episodes + 1),
        "Episodic Return": episode_rewards,
        "Scores": episode_scores,
        "Epsilon": episode_epsilons
    }).to_csv(f"data/{date}.csv", index=False)



