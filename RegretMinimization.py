import numpy as np
import numba

from utils import rand_choice_nb


@numba.experimental.jitclass([
    ('num_actions', numba.int32),
    ('regret_sum', numba.float64[:]),
    ('strategy', numba.float64[:]),
    ('strategy_sum', numba.float64[:]),
    ('opponent_strategy', numba.float64[:])
])
class RegretMinimization:
    """
    A python implementation of Regret Minimization with JIT-compilation using numba.

    Based on:
    An Introduction to Counterfactual Regret Minimization
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(self.num_actions)  # stores the regret sums for each action
        self.strategy = np.zeros(self.num_actions)  # stores the current strategy (a probability dist for each action)
        self.strategy_sum = np.zeros(self.num_actions)  # stores the sum of each past strategy

        # for testing: a simple static opponent strategy for sampling opponent actions
        self.opponent_strategy = np.random.rand(self.num_actions)

    def get_strategy(self, realization_weight=1):
        """
        Computes the new mixed-strategy using the current regret sums
        :returns strategy The computed strategy
        """
        normalizing_sum = 0

        # update the strategy using the regret sums and calculate the normalizing sum
        for i in range(self.num_actions):
            self.strategy[i] = self.regret_sum[i] if self.regret_sum[i] > 0 else 0
            normalizing_sum += self.strategy[i]

        # normalize the strategy array and calculate the strategy sums
        for i in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[i] /= normalizing_sum
            else:
                # normalizing sum could be non-positive -> make strategy uniform
                self.strategy[i] = 1.0 / self.num_actions

            # add the calculated strategy to the sum of every strategy to later on compute the average
            self.strategy_sum[i] += realization_weight * self.strategy[i]

        return self.strategy

    def get_action(self, strategy):
        """
        Samples an action with the current mixed-strategy
        :returns action An action sampled from the current strategy
        """
        return rand_choice_nb([i for i in range(self.num_actions)], self.strategy)

    def get_average_strategy(self):
        """
        Calculates the average strategy
        :returns avg_strategy The average strategy
        """
        avg_strategy = np.zeros(self.num_actions)
        normalizing_sum = np.sum(self.strategy_sum)

        # normalize the strategy sums, hence calculating the average strategy
        for i in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[i] = self.strategy_sum[i] / normalizing_sum
            else:
                # normalizing sum could be non-positive -> make strategy uniform
                avg_strategy[i] = 1.0 / self.num_actions

        return avg_strategy

    def test(self, iterations, utility_function):
        """
        Train the algorithm using RegretMatching for n iterations using the test opponent strategy
        :param iterations The training iterations
        :param utility_function The utility function for calculating the action utilities
        """
        for i in range(iterations):

            # sample action using the current mixed-strategy
            strategy = self.get_strategy()
            action = self.get_action(strategy)
            opponent_action = self.get_action(self.opponent_strategy)

            # compute action utilities using the specified utility function
            action_utility = utility_function(action, opponent_action)

            # accumulate actions regrets
            for j in range(self.num_actions):
                self.regret_sum[j] += action_utility[j] - action_utility[action]

    def train_step(self, action_utility):
        """
        Perform a single train step
        :param action_utility The pre-calculated action utilities
        """
        # sample action using the current mixed-strategy
        strategy = self.get_strategy()
        action = self.get_action(strategy)

        # accumulate actions regrets
        for j in range(self.num_actions):
            self.regret_sum[j] += action_utility[j] - action_utility[action]
