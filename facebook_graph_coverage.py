import random

from base_task import BaseTask
import numpy as np
import os
from typing import Set, List
import networkx as nx


class FacebookGraphCoverage(BaseTask):
    def __init__(self, budget: float, n: int = None, alpha=0.05, beta=1000, seed=0, graph_path: str = None, knapsack = True, prepare_max_pair = True, print_curvature=False,construct_graph = False, cost_mode="normal", graph_suffix = "", enable_packing = False, constraint_count = 4):
        """
        Inputs:
        - n: max_nodes
        - b: budget
        """
        super().__init__()
        
        if graph_path is None:
            raise Exception("Please provide a graph.")

        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.max_nodes = n
        # self.graph: nx.Graph = self.load_graph(graph_path + "/facebook_combined.txt")

        self.enable_packing_constraint = enable_packing
        self.cc = constraint_count

        # cost parameters
        self.graph_path = graph_path

        graph_name = "graph" + graph_suffix + ".txt"
        cost_name = "costs" + graph_suffix + ".txt"

        self.costs_obj = []

        if construct_graph:
            self.graph: nx.Graph = self.load_original_graph("./dataset/facebook/facebook_combined.txt")

            with open(self.graph_path + "/" + graph_name, "w") as f:
                for edge in self.graph.edges:
                    f.write(f"{edge[0]} {edge[1]}\n")

            self.nodes = list(self.graph.nodes)
            self.nodes = [int(node_str) for node_str in self.nodes]
            self.nodes.sort()
            self.objs = list(range(0, len(self.nodes)))

            self.assign_costs(knapsack, cost_mode)

            # with open(self.graph_path + "/" + cost_name, "w") as f:
            #     for node in range(0, self.max_nodes):
            #         f.write(f"{self.costs_obj[node]}\n")
        else:
            self.graph: nx.Graph = self.load_graph(graph_path + "/" + graph_name)
            self.nodes = list(self.graph.nodes)
            self.nodes = [int(node_str) for node_str in self.nodes]
            self.nodes.sort()
            self.objs = list(range(0, len(self.nodes)))

            with open(self.graph_path + "/" + cost_name, "r") as f:
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    self.costs_obj.append(float(line.rstrip("\n")))

        self.budget = budget

        if knapsack == False:
            self.costs_obj = [
                1
                for obj in self.objs
            ]

    @property
    def ground_set(self):
        return self.objs

    def calculate_m(self):
        ret = [self.cutout_marginal_gain(ele) for ele in self.ground_set]
        return ret

    def load_original_graph(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.txt does not exist.")
        intact_graph: nx.Graph = nx.read_edgelist(path)
        nodes = random.sample(list(intact_graph.nodes), min(len(list(intact_graph.nodes)), self.max_nodes))
        nodes.sort()
        # print(f"nodes:{nodes[:10]}")
        return intact_graph.subgraph(nodes)

    def load_graph(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.txt does not exist.")
        intact_graph: nx.Graph = nx.read_edgelist(path)

        return intact_graph

    def internal_objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        if type(S) == int:
            S = [S]
        neighbors = set([self.nodes[s] for s in S])
        for s in S:
            new_neighbors = set(self.graph.neighbors(str(self.nodes[s])))
            new_neighbors = set([int(x) for x in new_neighbors])
            neighbors = neighbors | new_neighbors

        return len(neighbors)

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        # print(f"s:{singleton}, l:{len(self.costs_obj)}")
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]


def main():
    model = FacebookGraphCoverage(budget=10.0, n=10000, graph_path="./dataset/facebook/facebook_combined.txt")

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
