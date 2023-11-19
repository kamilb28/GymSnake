from rl import Qlearn, get_state
from evns.GymSnakeSimpleObsEnv import SnakeSimpleObsEnv
import pickle

env = SnakeSimpleObsEnv(render_mode="human", size=10)
qlearn = Qlearn(alpha=0.1, gamma=0.99)

pickle_in = open('learn.pkl', 'rb')
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
