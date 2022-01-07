import numpy as np
from numba import jit
import numba


@numba.experimental.jitclass([
    ('num_actions', numba.int32),
    ('regretSum', numba.float64[:]),
    ('strategy', numba.float64[:]),
    ('strategySum', numba.float64[:]),
    ('opponentStrategy', numba.float64[:])
])
class RegretMinimization:
    """
    A python implementation of Regret Minimization using JIT-compilation through numba.

    Based on:
    An Introduction to Counterfactual Regret Minimization
    (http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
    """
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regretSum = np.zeros(self.num_actions)
        self.strategy = np.zeros(self.num_actions)
        self.strategySum = np.zeros(self.num_actions)

        # for testing, a simple static opponent strategy for sampling opponent actions
        self.opponentStrategy = np.random.rand(self.num_actions)

    def get_strategy(self):
        """
        Computes the new mixed-strategy using the current regretSums
        :returns strategy The computed strategy
        """
        normalizingSum = 0

        # update the strategy using the regret sums and calculate the normalizing sum
        for i in range(self.num_actions):
            self.strategy[i] = self.regretSum[i] if self.regretSum[i] > 0 else 0
            normalizingSum += self.strategy[i]

        # normalize the strategy list and calculate the strategy sums
        for i in range(self.num_actions):
            if normalizingSum > 0:
                self.strategy[i] /= normalizingSum
            else:
                # normalizing sum could be non-positive -> make strategy uniform
                self.strategy[i] = 1.0 / self.num_actions

            self.strategySum[i] += self.strategy[i]
        return self.strategy

    def get_action(self, strategy):
        """
        Samples an action with the current mixed-strategy
        :returns action An action sampled from the current strategy
        """
        return [i for i in range(self.num_actions)][np.searchsorted(np.cumsum(strategy), np.random.random(),
                                                                    side="right")]

    def get_average_strategy(self):
        """
        Calculates the average strategy
        :returns avgStrategy The average strategy
        """
        avgStrategy = np.zeros(self.num_actions)
        normalizingSum = np.sum(self.strategySum)

        for i in range(self.num_actions):
            if normalizingSum > 0:
                avgStrategy[i] = self.strategySum[i] / normalizingSum
            else:
                avgStrategy[i] = 1.0 / self.num_actions
        return avgStrategy

    def train(self, iterations, utility_function):
        """
        Train the algorithm using RegretMatching
        :param iterations The training iterations
        :param utility_function The utility function for calculating the action utilities
        """
        actionUtility = np.zeros(self.num_actions)
        for i in range(iterations):

            # sample action using the current mixed-strategy
            strat = self.get_strategy()
            action = self.get_action(strat)
            opponentAction = self.get_action(self.opponentStrategy)

            # compute action utilities using the passed utility function
            actionUtility = utility_function(action, opponentAction)

            # accumulate actions regrets
            for j in range(self.num_actions):
                self.regretSum[j] += actionUtility[j] - actionUtility[action]

    def train_step(self, action_utility):
        """
        Perform a single train step
        :param action_utility The pre-calculated action utilities
        """
        # sample action using the current mixed-strategy
        strat = self.get_strategy()
        action = self.get_action(strat)

        # accumulate actions regrets
        for j in range(self.num_actions):
            self.regretSum[j] += action_utility[j] - action_utility[action]
