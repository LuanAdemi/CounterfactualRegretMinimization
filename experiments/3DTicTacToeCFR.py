from CFR import CFR
import numpy as np
from copy import deepcopy

import gym
from utils import EnvWrapper
from gym_tictactoe_np.envs.tictactoe_np_env import TicTacToeEnv


class ThreeDTicTacToeEnv(EnvWrapper):
    def __init__(self):
        self.env = TicTacToeEnv()

    @property
    def done(self):
        return self.env.done

    def perform(self, action):
        new_env = TicTacToeEnv()
        new_env.board = self.env.board.copy()
        new_env.step(action)
        return new_env

    def get_actions(self):
        return self.env.get_available_actions(self.env.board)

    def payoff(self, player):
        return 1 if self.env.done and self.env.current == player else -1

    def render(self):
        return self.env.render()


env = ThreeDTicTacToeEnv()
env2 = env.perform([0, 0, 0])
print(env.render(), env2.render())
