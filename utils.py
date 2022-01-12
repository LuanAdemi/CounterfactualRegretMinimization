import numba
import numpy as np


@numba.njit()
def rand_choice_nb(arr, prob):
    """
    See https://github.com/numba/numba/issues/2539
    This also normalizes the probabilities in case they are non-probabilistic.
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob/np.sum(prob)), np.random.random(), side="right")]


@numba.njit()
def get_action_from_dist(strategy):
    """
    Samples an action using the given mixed-strategy
    :returns action An action sampled from the current strategy
    """
    return rand_choice_nb([i for i in range(len(strategy))], strategy)


class EnvWrapper:
    @property
    def done(self):
        """
        Returns true, if the last actions resulted in a terminal state
        :return: true, if state terminal, else false
        """
        return NotImplementedError

    def perform(self, action):
        """
        Performs an action
        :param action: The action which will be performed
        :return: The next state
        """
        return NotImplementedError

    def get_actions(self):
        """
        Returns a list of legal actions in the current state
        :return: A list of actions
        """
        return NotImplementedError

    def payoff(self, player):
        """
        Returns the payoff for the specified player in a terminal state
        (chess for example, -1 if loss, 0 if draw, 1 if win)
        :param player: The current player
        :return: The payoff for the player
        """
        return NotImplementedError
