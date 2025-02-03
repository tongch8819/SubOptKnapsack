from abc import ABC
import networkx as nx
import random
from typing import List, Set
from copy import deepcopy
import matplotlib.pyplot as plt


class Constraint:
    def __init__(self, name="general constraint"):
        self.name = name

class CardinalityConstraint(Constraint):
    def __init__(self, k: int):
        super().__init__("cardinality constraint")
        self.size = k

class KnapsackConstraint(Constraint):
    def __init__(self, b: float):
        super().__init__("knapsack constraint")
        self.budget = b



class Objective:
    def __init__(self):
        self.ground_set = None
        self.is_monotone = True
        self.total_curvature = None

    def eval(self, S):
        pass

    def marginal_gain(self, single: int, base: List[int]):
        if len(base) == 0:
            base2 = [single]
        else:
            base = list(base)
            base2 = list(set(base + [single]))
        fS1 = self.eval(base)
        fS2 = self.eval(base2)
        res = fS2 - fS1
        if self.is_monotone:
            assert res >= 0., f"f({base2}) - f({base}) = {fS2:.2f} - {fS1:.2f}"
        return res

    def cutout_marginal_gain(self, singleton: int, base: Set[int]):
        """
        return f(singleton | base - singleton)
        """
        if type(base) is not set:
            base = set(base)
        assert singleton in base
        # assert type(base) is set, "{} is not set".format(type(base))
        base2 = deepcopy(base)
        base2.remove(singleton)  # no return value
        fS2, fS1 = self.eval(base), self.eval(base2)
        res = fS2 - fS1
        assert res >= 0., f"f({base}) - f({base2}) = {fS2:.2f} - {fS1:.2f}\n{base - base2}"
        return res
    
    def gen_total_curvature(self):
        if self.total_curvature is not None:
            return self.total_curvature

        min_ratio = 1
        for e in self.ground_set:
            nume = self.cutout_marginal_gain(e, self.ground_set)
            denom = self.objective([e])
            assert denom >= nume
            ratio = nume / (denom + 1e-7)
            min_ratio = min(min_ratio, ratio)
        self.total_curvature = 1 - min_ratio
        return self.total_curvature

class VertexCutObjective(Objective):

    def __init__(self, graph=None):
        if graph is None:
            # use a default weighted directed graph
            # Parameters
            num_nodes = 10  # Number of vertices
            num_edges = 30  # Number of edges

            # Create a complex weighted undirected graph
            G = self._create_complex_weighted_graph(num_nodes, num_edges)
            self.graph = G
        else:
            self.graph = graph

        self.ground_set = list(range(1, num_nodes + 1))
        self.is_monotone = False
        self.total_curvature = None
    
    def eval(self, S):
        return self._compute_edge_weight_sum(self.graph, S)
    
    def _compute_edge_weight_sum(self, G, S):
        """
        Compute the sum of edge weights between the vertex subset S and its complement in graph G.

        Parameters:
        G (networkx.Graph): The weighted undirected graph.
        S (set): A subset of vertices in G.

        Returns:
        float: The sum of edge weights between S and its complement.
        """
        # Initialize the sum
        edge_weight_sum = 0

        # Iterate over all edges in the graph
        for u, v, data in G.edges(data=True):
            # Check if one vertex is in S and the other is not
            if (u in S and v not in S) or (u not in S and v in S):
                # Add the edge weight to the sum
                edge_weight_sum += data.get('weight', 1)  # Default weight is 1 if not specified

        return edge_weight_sum
    

    def _create_complex_weighted_graph(self, num_nodes, num_edges, min_weight=1, max_weight=10):
        """
        Create a weighted undirected graph with a specified number of nodes and edges.
        
        Parameters:
        num_nodes (int): The number of vertices in the graph.
        num_edges (int): The number of edges in the graph.
        min_weight (int): Minimum edge weight.
        max_weight (int): Maximum edge weight.

        Returns:
        G (networkx.Graph): A weighted undirected graph.
        """
        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        G.add_nodes_from(range(1, num_nodes + 1))

        # Add edges with random weights until we have the desired number of edges
        while len(G.edges()) < num_edges:
            # Randomly select two distinct nodes
            u, v = random.sample(range(1, num_nodes + 1), 2)
            # Add an edge with a random weight if the edge doesn't exist already
            if not G.has_edge(u, v):
                weight = random.randint(min_weight, max_weight)
                G.add_edge(u, v, weight=weight)

        return G
    
    def _visualize_graph(self, G):
        """
        Visualize the weighted graph using Matplotlib.

        Parameters:
        G (networkx.Graph): The graph to visualize.
        """
        pos = nx.spring_layout(G)  # Position nodes using a force-directed layout
        edge_labels = nx.get_edge_attributes(G, 'weight')

        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Weighted Undirected Graph")
        plt.show()
    

class ProblemInstance(ABC):
    def __init__(self, obj=None, constraint=None):
        self.objective = obj
        self.constraint = constraint

    