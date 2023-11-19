import numpy as np
from tqdm import tqdm
from evns.GymSnakeSimpleObsEnv import SnakeSimpleObsEnv
from evns.GymSnakeEnv import ACTION_SPACE
import pickle

class Qlearn:
    Q = {}

    def __init__(self, alpha, gamma):
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


if __name__ == '__main__':
    num_of_episodes = 100000
    eps = 1.  # exploration parameter

    env = SnakeSimpleObsEnv(render_mode=None, size=10)
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

        # decrease epsilon over time (in halfway selection strategy will be almost entirely greedy)
        eps = eps - (1.1/num_of_episodes) if eps > 0.05 else 0.05

        # Update progress bar
        progress_bar.set_postfix(epsilon=f'{eps:.2f}',
                                 reward=f'{episodic_reward}')
        progress_bar.update(1)


    env.close()

    f = open("learn.pkl", "wb")
    pickle.dump(qlearn.Q, f)
    f.close()
