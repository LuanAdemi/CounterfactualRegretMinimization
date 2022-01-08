import numba
import numpy as np
from RegretMinimization import RegretMinimization

"""
A little example of training two regret-minimizing agents to play rock-paper-scissors.
This leads to a nash-equilibrium with the optimal policy being a probability of 1/3 for each action.
"""

# Training epochs
EPOCHS = 100000


@numba.jit(nopython=True)
def utility_function(action, opponent_action):
    """
    Calculates the action utilities using the given actions
    :returns action_utility The pre-calculated action utilities
    """
    action_utility = np.zeros(3)
    action_utility[opponent_action] = 0
    action_utility[0 if opponent_action == 3 - 1 else opponent_action + 1] = 1
    action_utility[3 - 1 if opponent_action == 0 else opponent_action - 1] = -1
    return action_utility


@numba.jit(nopython=True)
def train_agents():
    """
    Trains two agents to play RPS against each other. This will optimally result in a nash equilibrium
    :returns average_strategies The average_strategy of each agent over EPOCHS.
    """
    # create two regret-minimizing agents
    agent1 = RegretMinimization(3)
    agent2 = RegretMinimization(3)

    # main loop
    for e in numba.prange(EPOCHS):
        # sample actions
        strategy_1 = agent1.get_strategy()
        action_1 = agent1.get_action(strategy_1)

        strategy_2 = agent2.get_strategy()
        action_2 = agent2.get_action(strategy_2)

        # get the action utilities
        action_utility_1 = utility_function(action_1, action_2)
        action_utility_2 = utility_function(action_2, action_1)

        agent1.train_step(action_utility_1)
        agent2.train_step(action_utility_2)

    return agent1.get_average_strategy(), agent2.get_average_strategy()


@numba.njit(parallel=True)
def train_n_agent_pairs(n):
    """
    Uses numba's auto parallelization framework to execute multiple training routines simultaneously.
    """
    for i in numba.prange(n):
        print("Worker #" + str(i) + ": ", train_agents())


# play 16 parallel games
train_n_agent_pairs(16)
