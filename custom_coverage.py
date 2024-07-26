import random
import time

from base_task import BaseTask
import numpy as np
import os
from typing import Set, List
import networkx as nx


class CustomCoverage(BaseTask):
    def __init__(self, budget: float, n: int = None, alpha=0.2, beta=0.1, min_cost=0.1, gamma=0.9, seed=60,
                 graph_path: str = None, knapsack=True, prepare_max_pair=True, print_curvature=False, construct_graph = False):
        """
        Inputs:
        - n: max_nodes
        - b: budget
        """
        super().__init__()
        if graph_path is None:
            raise Exception("Please provide a graph.")

        if construct_graph:
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)

            np.random.seed(seed)
            random.seed(seed)

            self.max_nodes = n
            # custom parameters
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

            self.graph_path = graph_path

            strategies = ["additive", "shared", "multi_cluster"]

            s = 2

            if s == 0:
                self.beta = max(beta, max(2 * alpha/(1-alpha), 0))

            self.strategy = strategies[s]

            prepare = eval("self.prepare_graph_" + self.strategy)

            prepare()

            # write cardinality
            if knapsack:
                self.costs_obj = [
                    (min_cost + random.random()) * 4
                    for node in range(0, self.max_nodes)
                ]
            else:
                self.costs_obj = [
                    1
                    for node in range(0, self.max_nodes)
                ]

            with open(self.graph_path + "/" + "costs.txt", "w") as f:
                for node in range(0, self.max_nodes):
                    f.write(f"{self.costs_obj[node]}\n")

        self.graph_path = graph_path

        self.graph: nx.Graph = self.load_graph(graph_path + "/" + "graph.txt")

        self.nodes = list(self.graph.nodes)
        self.nodes = [int(node_str) for node_str in self.nodes]
        self.objs = list(range(0, len(self.nodes)))

        self.budget = budget

        self.costs_obj = []

        with open(self.graph_path + "/" + "costs.txt", "r") as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                self.costs_obj.append(float(line.rstrip("\n")))

        if prepare_max_pair:
            self.prepare_max_2_pair()

        if print_curvature:
            self.print_curvature()

    # First, c2 should be great
    # Second, density should decrease rapidly

    def prepare_graph_additive(self):
        # nodes = np.array(range(0, self.max_nodes))

        total_nodes = list(range(0, self.max_nodes))

        nodes = random.sample(total_nodes, int(self.gamma * len(total_nodes)))

        unvisited_nodes = nodes
        visited_cluster = None

        edges = []

        prev_neighbor_count = 0.

        # generation strategy:
        # A new node must have at least random(alpha, beta) * len(unvisited nodes) neighbors in unvisited nodes
        # A new node should not have more nodes than previous nodes

        for i in range(0, self.max_nodes):
            # first, choose
            # pick = int((random.random() * (self.beta - self.alpha) + self.alpha) * len(unvisited_nodes))

            pick = int(random.uniform(0.5, self.alpha) * len(unvisited_nodes))

            nodes_selected = random.sample(unvisited_nodes, pick)

            for node in nodes_selected:
                unvisited_nodes.remove(node)

            if i > 0:
                pick_existing = int(random.uniform(0.5, self.beta) * pick)
                pick_existing = min(pick_existing, len(visited_cluster))
                if pick_existing > 0:
                    existing_selected = random.sample(visited_cluster, pick_existing)
                    nodes_selected = nodes_selected + existing_selected
            else:
                visited_cluster = nodes_selected

            visited_cluster = list(set(visited_cluster) | set(nodes_selected))

            edges.append(nodes_selected)

        with open(self.graph_path + "/o-graph.txt", "w") as f:
            for node in range(0, self.max_nodes):
                for edge in edges[node]:
                    f.write(f"{node} {edge}\n")

    def prepare_graph_shared(self):
        # nodes = np.array(range(0, self.max_nodes))

        total_nodes = list(range(0, self.max_nodes))

        shared_nodes = random.sample(total_nodes, int(self.gamma*len(total_nodes)))

        private_nodes = list(set(total_nodes) - set(shared_nodes))

        edges = []

        current_shared = int(self.beta*  len(shared_nodes))

        step_length = int(current_shared/(self.max_nodes-1))

        for i in range(0, self.max_nodes):
            # first, choose
            # pick = int((random.random() * (self.beta - self.alpha) + self.alpha) * len(unvisited_nodes))

            pick = int(self.alpha * len(private_nodes))

            nodes_selected = list(random.sample(private_nodes, pick))

            pick_existing = current_shared # int(self.beta * len(shared_nodes))

            if pick_existing > 0:
                existing_selected = random.sample(shared_nodes, pick_existing)
                nodes_selected = nodes_selected + existing_selected

            if nodes_selected.count(i) > 0:
                nodes_selected.remove(i)
            # prev_neighbor_count = len(nodes_selected)

            edges.append(nodes_selected)

            current_shared -= step_length

        with open(self.graph_path + "/graph.txt", "w") as f:
            for node in range(0, self.max_nodes):
                for edge in edges[node]:
                    f.write(f"{node} {edge}\n")

    def prepare_graph_multi_cluster(self):
        # nodes = np.array(range(0, self.max_nodes))
        cluster_count = 6

        total_nodes = list(range(0, self.max_nodes))

        shared_nodes = random.sample(total_nodes, int(self.gamma * len(total_nodes)))

        cluster_size = int(len(shared_nodes)/cluster_count)

        clusters = []

        for i in range(0, cluster_count):
            if i < cluster_count - 1:
                clusters.append(shared_nodes[cluster_size*i:cluster_size*(i+1)])
            else:
                clusters.append(shared_nodes[cluster_size*i:])

        private_nodes = list(set(total_nodes) - set(shared_nodes))

        edges = []

        prev_neighbor_count = 0.

        # generation strategy:
        # A new node must have at least random(alpha, beta) * len(unvisited nodes) neighbors in unvisited nodes
        # A new node should not have more nodes than previous nodes

        for i in range(0, self.max_nodes):
            # first, choose
            # pick = int((random.random() * (self.beta - self.alpha) + self.alpha) * len(unvisited_nodes))
            cluster_id = i % cluster_count
            candidate_nodes = clusters[cluster_id]

            pick = int(len(private_nodes) * random.uniform(0.01, self.alpha))

            nodes_selected = list(random.sample(private_nodes, pick))

            pick_existing = int(len(candidate_nodes) * random.uniform(0.1, self.beta))

            if pick_existing > 0:
                existing_selected = random.sample(candidate_nodes, pick_existing)
                nodes_selected = nodes_selected + existing_selected

            if nodes_selected.count(i) > 0:
                nodes_selected.remove(i)
            # prev_neighbor_count = len(nodes_selected)

            edges.append(nodes_selected)

        with open(self.graph_path + "/o-graph.txt", "w") as f:
            for node in range(0, self.max_nodes):
                for edge in edges[node]:
                    f.write(f"{node} {edge}\n")

    @property
    def ground_set(self):
        return self.objs

    def load_graph(self, path: str):
        if not os.path.isfile(path):
            raise OSError(f"File {path} does not exist.")
        intact_graph: nx.Graph = nx.read_edgelist(path)
        nodes = list(intact_graph.nodes)

        return intact_graph.subgraph(nodes)

    def internal_objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        neighbors = set([self.nodes[s] for s in S])

        for s in S:
            neighbors = neighbors | set(self.graph.neighbors(str(self.nodes[s])))
        return 1000 * len(neighbors) / len(self.nodes)

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]


def main():
    model = CustomCoverage(budget=10.0, n=10000, graph_path="./dataset/facebook/facebook_combined.txt")

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
