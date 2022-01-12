from copy import deepcopy
from utils import EnvWrapper
from gym_tictactoe_np.envs.tictactoe_np_env import TicTacToeEnv

class ThreeDTicTacToeEnv(EnvWrapper):
    def __init__(self):
        self.env = TicTacToeEnv()

    @property
    def done(self):
        return self.env.done

    def perform(self, action):
        new_env = deepcopy(self)
        new_env.step(self.env.get_available_actions(self.env.board)[action])
        return new_env

    def get_actions(self):
        actions = self.env.get_available_actions(self.env.board)
        return [i for i in range(len(actions))]

    def payoff(self, player):
        return 1 if self.env.done and self.env.current == player else -1

    def render(self):
        return self.env.render()

    def step(self, action):
        return self.env.step(action)
