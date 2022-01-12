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


class GymEnvWrapper:
    def __init__(self, env_name):
        self.env_name = env_name

    # TODO
