import numba
import numpy as np

from RegretMinimization import RegretMinimization
from envs import ThreeDTicTacToeEnv

regret_minimization_type = numba.deferred_type()
regret_minimization_type.define(RegretMinimization.class_type.instance_type)


@numba.experimental.jitclass([
    ('realization_weight', numba.float64),
    ('num_actions', numba.int32),
    ('info_set', numba.types.unicode_type),
    ('agent', regret_minimization_type)
])
class Node:
    """
    An information set node for CFR containing the regret.
    """
    def __init__(self, num_actions, info_set=""):
        self.num_actions = num_actions
        self.agent = RegretMinimization(self.num_actions)
        self.info_set = info_set  # a string representation of the information node

    def __str__(self):
        """
        A string representation for the information node
        :return: repr: A string containing the current info_set and the average strategy of the information node
        """
        return f"{self.info_set}: {self.get_average_strategy()}"

    """
    Forward the attributes of the regret minimizing agent
    """
    @property
    def regret_sum(self):
        return self.agent.regret_sum

    @property
    def strategy(self):
        return self.agent.strategy

    @property
    def strategy_sum(self):
        return self.agent.strategy_sum

    def set_info_set(self, info_set):
        """
        Sets the current info_set
        :param info_set
        """
        self.info_set = info_set

    def get_strategy(self, realization_weight):
        """
        Get the current strategy of the information node
        :param realization_weight
        :returns strategy: The new strategy
        """
        return self.agent.get_strategy(realization_weight)

    def get_average_strategy(self):
        """
        Get the average strategy of the information node
        :returns avg_strategy: The average strategy
        """
        return self.agent.get_average_strategy()


# TODO: Numba this. It might be hard...
def cfr(nodeMap, env, history, realization_weights):
    """
    The implementation of CFR using recursion.

    Based on:
    An Introduction to Counterfactual Regret Minimization

    Traverses each node of the game tree and calculates their utility. We later pick the nodes
    with the highest utility and choose the corresponding action.

    :param nodeMap: The dict containing the whole game tree
    :param env: The current information state (basically the state of the env)
    :param history: The action history
    :param realization_weights: The probabilities of playing the current information set for each player (tuple)
    :returns node_util: The utility of the node
    """
    turns = len(history)
    num_agents = len(realization_weights)
    current_agent = turns % num_agents
    info_set = str(env) + str(history)

    possible_actions = env.get_actions()
    num_actions = len(possible_actions)

    # return payoff for terminal states (chess for example, -1 if loss, 0 if draw, 1 if win)
    if env.done:
        # if this is a terminal node, return the payoff for the current player
        return env.payoff(current_agent)

    # get information node for the current information set
    node = nodeMap[info_set] if info_set in nodeMap else Node(num_actions, info_set)
    nodeMap[info_set] = node

    # calculate the current strategy
    strategy = node.get_strategy(realization_weights[current_agent])
    util = np.zeros(num_actions)

    node_util = 0

    # perform cfr on each game node below this one
    for i, a in enumerate(possible_actions):
        # get the new state and history by performing the action
        new_env = env.perform(a)
        new_history = history + str(a)

        # update the realization weights
        new_realization_weights = realization_weights
        new_realization_weights[current_agent] *= strategy[i]

        # get the util for the new game node
        util[i] = -cfr(nodeMap, new_env, new_history, new_realization_weights)
        node_util += strategy[i] * util[i]

    # calculate the regret sum for each action
    for i, a in enumerate(possible_actions):
        regret = util[i] - node_util
        node.regret_sum[i] += realization_weights[current_agent] * regret

    return node_util


def train(env, iterations):
    util = 0
    nodeMap = {}
    for i in range(iterations):
        util += cfr(nodeMap, env, "", [1., 1.])
    print("Average game value: " + util / iterations)

    for n in nodeMap.values():
        print(n)
