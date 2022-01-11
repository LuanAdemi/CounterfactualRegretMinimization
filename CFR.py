import numba
import numpy as np

from RegretMinimization import RegretMinimization


@numba.experimental.jitclass([
    ("num_actions", numba.int32),
    ("realization_weight", numba.float32)
])
class Node:
    """
    An information set node for CFR containing the regret,
    """
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.agent = RegretMinimization(self.num_actions)
        self.info_set = ""

    def __str__(self):
        """
        A string representation for the information node
        :return: repr A string containing the current info_set and the average strategy of the information node
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

    def get_strategy(self, realization_weight):
        """
        Get the current strategy of the information node
        :param realization_weight
        """
        return self.agent.get_strategy(realization_weight)

    def get_average_strategy(self):
        """
        Get the average strategy of the information node
        """
        return self.agent.get_average_strategy()


class CFR:
    def __init__(self, env):
        self.nodeMap = {}
        self.env = env
        self.num_actions = self.env.action_space

    def cfr(self, state, history, agents):
        turns = len(history)
        num_agents = len(agents)
        current_agent = turns % num_agents
        info_set = str(state) + str(history)

        # return payoff for terminal states
        # TODO

        # get information node for the current information set
        node = self.nodeMap[info_set]

        # if the node doesn't exist, create it
        if node is None:
            node = Node()
            node.info_set = info_set
            self.nodeMap[info_set] = node

        strategy = node.get_strategy(agents[current_agent])
        util = np.zeros(self.num_actions)
        node_util = 0

        actions = self.env.load(state)

        for a in actions:
            new_env = state.copy()
