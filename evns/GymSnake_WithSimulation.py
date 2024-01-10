from evns.GymSnakeEnv import SnakeEnv
from evns.GymSnakeSimpleObsEnv import SnakeSimpleObsEnv


class SnakeEnvWithSimulation(SnakeSimpleObsEnv):

    def __init__(self, render_mode=None, size=10, import_board=None):
        super().__init__(render_mode, size, import_board)

    def step(self, action, simulation=False):
        if not simulation:
            super().step(action)
            return
        #todo

    def get_copy(self):
        return SnakeEnvWithSimulation()