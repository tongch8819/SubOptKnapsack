import random

from base_task import BaseTask
import numpy as np
import os
from typing import Set, List
import networkx as nx


class FacebookGraphCoverage(BaseTask):
    def __init__(self, budget: float, n: int = None, alpha = 0.05, beta=1000, graph_path: str = None, knapsack = True, prepare_max_pair = True, print_curvature=False,construct_graph = False, cost_mode="normal", graph_suffix = ""):
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
        # self.graph: nx.Graph = self.load_graph(graph_path + "/facebook_combined.txt")

        min_cost = 0.4
        factor = 4

        # cost parameters
        self.graph_path = graph_path

        graph_name = "graph" + graph_suffix + ".txt"
        cost_name = "costs" + graph_suffix + ".txt"

        # if construct_graph:
        #
        # else:
        #     if knapsack:
        #         if cost_mode == "normal":
        #             self.costs_obj = [
        #                 # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
        #                 (min_cost + random.random()) * factor
        #                 for node in self.nodes
        #             ]
        #         elif cost_mode == "small":
        #             self.costs_obj = [
        #                 # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
        #                 max(min_cost, random.gauss(mu=min_cost, sigma=1) * factor)
        #                 for node in self.nodes
        #             ]
        #         elif cost_mode == "big":
        #             self.costs_obj = [
        #                 # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
        #                 max(min_cost, min(min_cost + 1, random.gauss(mu=min_cost + 1, sigma=1) * factor))
        #                 for node in self.nodes
        #             ]
        #     else:
        #         self.costs_obj = [
        #             1
        #             for node in self.nodes
        #         ]

        self.costs_obj = []

        if construct_graph:
            self.graph: nx.Graph = self.load_original_graph(graph_path + "/facebook_combined.txt")

            with open(self.graph_path + "/" + graph_name, "w") as f:
                for edge in self.graph.edges:
                    f.write(f"{edge[0]} {edge[1]}\n")

            self.nodes = list(self.graph.nodes)
            self.nodes = [int(node_str) for node_str in self.nodes]
            self.objs = list(range(0, len(self.nodes)))

            if knapsack:
                # self.objs.sort(key=lambda x: len(self.nodes[x]), reverse=True)
                if cost_mode == "normal":
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        (min_cost + random.random()) * factor
                        for node in self.nodes
                    ]
                elif cost_mode == "small":
                    cost_name = "small_costs.txt"
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        max(min_cost, random.gauss(mu=min_cost, sigma=1) * factor)
                        for node in self.nodes
                    ]
                elif cost_mode == "big":
                    cost_name = "big_costs.txt"
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        min(min_cost + 1, random.gauss(mu=min_cost + 1, sigma=1) * factor)
                        for node in self.nodes
                    ]
            else:
                # cardinality
                self.costs_obj = [
                    1
                    for node in self.nodes
                ]
            with open(self.graph_path + "/" + cost_name, "w") as f:
                for node in range(0, self.max_nodes):
                    f.write(f"{self.costs_obj[node]}\n")
        else:
            self.graph: nx.Graph = self.load_graph(graph_path + "/" + graph_name)
            self.nodes = list(self.graph.nodes)
            self.nodes = [int(node_str) for node_str in self.nodes]
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

        if prepare_max_pair:
            self.prepare_max_2_pair()

        if print_curvature:
            self.print_curvature()


    @property
    def ground_set(self):
        return self.objs

    def load_original_graph(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.txt does not exist.")
        intact_graph: nx.Graph = nx.read_edgelist(path)
        nodes = random.sample(list(intact_graph.nodes), min(len(list(intact_graph.nodes)), self.max_nodes))

        return intact_graph.subgraph(nodes)

    def load_graph(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.txt does not exist.")
        intact_graph: nx.Graph = nx.read_edgelist(path)

        return intact_graph

    def objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        neighbors = set([self.nodes[s] for s in S])
        for s in S:
            new_neighbors = set(self.graph.neighbors(str(self.nodes[s])))
            new_neighbors = set([int(x) for x in new_neighbors])
            neighbors = neighbors | new_neighbors

        return 10 * len(neighbors) / len(self.nodes)

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
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