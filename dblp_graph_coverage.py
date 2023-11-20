import random

from base_task import BaseTask
import numpy as np
import os
from typing import Set, List
import networkx as nx


class DblpGraphCoverage(BaseTask):
    def __init__(self, budget: float, n: int = None, alpha = 0.05, beta=10000, graph_path: str = None):
        """
        Inputs:
        - n: max_nodes
        - b: budget
        """
        if graph_path is None:
            raise Exception("Please provide a graph.")
        np.random.seed(1)
        random.seed(1)
        self.max_nodes = n
        self.graph: nx.Graph = self.load_graph(graph_path)

        self.nodes = list(self.graph.nodes)
        self.nodes = [int(node_str) for node_str in self.nodes]
        self.objs = list(range(0, len(self.nodes)))

        # cost parameters
        self.alpha = alpha
        self.beta = beta
        self.costs_obj = [
            self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
            for node in self.nodes
        ]
        self.b = budget

    @property
    def ground_set(self):
        return self.objs

    def load_graph(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.txt does not exist.")
        intact_graph: nx.Graph = nx.read_adjlist(path)
        nodes = random.sample(list(intact_graph.nodes), self.max_nodes)

        return intact_graph.subgraph(nodes)

    def objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        neighbors = set([self.nodes[s] for s in S])
        for s in S:
            neighbors = neighbors | set(self.graph.neighbors(str(self.nodes[s])))
        return len(neighbors) / len(self.nodes)

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]


def main():
    model = DblpGraphCoverage(budget=10.0, n=10000, graph_path="./dataset/com-dblp/com-dblp.top5000.cmty.txt")

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
