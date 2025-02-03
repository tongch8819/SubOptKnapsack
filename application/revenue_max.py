from base_task import BaseTask
import random
from typing import Set, List
import numpy as np
import pickle
from collections import defaultdict
import os


class RevenueMax(BaseTask):
    """
    Select a set of users in a social network to advertise a product to maximize the revenue.

    Model social network as undirected graph.

    The objective is a non-monotone submodular function.
    """

    def __init__(self, budget: float, m: int = None, edges: List[List[int]] = None, pckl_path: str = None, seed=1):
        """
        Inputs:
        - m: number of nodes
        """
        random.seed(seed)
        if pckl_path is None:
            if m is None or edges is None:
                raise ValueError("Missing Parameters.")
            self.num_nodes = m
            self.num_edges = len(edges)
        else:
            if not os.path.exists(pckl_path):
                raise OSError("Path is not valid.")
            with open(pckl_path, "rb") as rd:
                params = pickle.load(rd)
            self.num_nodes = params['num_vertices']
            edges = params['edges']
            self.num_edges = len(params['edges'])

        self.objects = [i for i in range(self.num_nodes)]
        self.weights = np.zeros(shape=(self.num_nodes, self.num_nodes))
        self.adjacent_table = [[] for _ in range(self.num_nodes)]
        self.adjacency = defaultdict(set)
        for u, v in edges:
            w = random.random()
            self.adjacent_table[u].append((v, w))
            self.adjacent_table[v].append((u, w))
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
            self.weights[u][v] = w
            self.weights[v][u] = w

        self.costs_obj = [self._compute_cost(u) for u in self.objects]

        self.b = budget

    @property
    def ground_set(self):
        return self.objects

    def _compute_cost(self, u):
        t = 0.
        mu = 0.2
        for v, w in self.adjacent_table[u]:
            t += w
        return 1 - np.exp(- mu * np.sqrt(t))

    def objective(self, S):
        S_set = set(S)
        N_set = set(self.ground_set)
        S_complement = N_set - S_set
        res = 0.
        for v in S_complement:
            partial_sum = 0.
            for u in S:
                if u in self.adjacency[v]:
                    partial_sum += self.weights[u][v]
            res += np.sqrt(partial_sum)
        return res

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]


def main():
    model = RevenueMax(budget=1.0, pckl_path="dataset/revenue/25_youtube_top5000.pkl")

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
