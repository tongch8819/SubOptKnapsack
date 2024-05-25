import math
import random

import pandas as pd
from scipy.stats import entropy

from base_task import BaseTask
import numpy as np
import os
from typing import Set, List
import networkx as nx

class AdultIncomeFeatureSelection(BaseTask):
    def __init__(self, budget: float, n: int = None, data_path: str = None, knapsack=True, seed = 0,
                 prepare_max_pair=True, print_curvature=False, construct_graph = False, min_cost = 0.4, factor = 4.0, cost_mode="normal"):
        """
        Inputs:
        - n: max_nodes
        - b: budget
        """
        super().__init__()
        if data_path is None:
            raise Exception("Please provide a graph.")
        np.random.seed(seed)
        random.seed(seed)

        self.samples = []

        # printed = False

        with open(data_path + "/" + "binary_data.txt", "r") as f:
            while True:
                line = f.readline()
                if line != "":
                    sample = list(line.rstrip("\n").split(" "))
                    sample.remove("")
                    # if not printed:
                    #     print(line)
                    #     print("*************")
                    #     print(sample[112])
                    #     print("**********")
                    #     printed = True
                    self.samples.append(sample)
                else:
                    break

        self.samples = random.sample(self.samples, n)

        self.samples = np.array(self.samples)

        self.objs = list(range(0, self.samples.shape[1] - 1))
        self.feature_count = len(self.objs)

        self.x = self.samples[:, :self.feature_count]
        self.y = self.samples[:, self.feature_count]

        self.y = self.y.astype(float)

        self.costs_obj = []

        self.total_samples = len(self.samples)
        cost_name = "costs.txt"

        # cost parameters
        if construct_graph:
            if knapsack:
                # self.objs.sort(key=lambda x: len(self.nodes[x]), reverse=True)
                if cost_mode == "normal":
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        (min_cost + random.random()) * factor
                        for node in range(0, self.feature_count)
                    ]
                elif cost_mode == "small":
                    cost_name = "small_costs.txt"
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        max(min_cost, random.gauss(mu=min_cost, sigma=1) * factor)
                        for node in range(0, self.feature_count)
                    ]
                elif cost_mode == "big":
                    cost_name = "big_costs.txt"
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        min(min_cost + 1, random.gauss(mu=min_cost + 1, sigma=1) * factor)
                        for node in range(0, self.feature_count)
                    ]
            else:
                # cardinality
                self.costs_obj = [
                    1
                    for node in range(0, self.feature_count)
                ]
            with open(data_path + "/" + cost_name, "w") as f:
                for node in range(0, self.feature_count):
                    f.write(f"{self.costs_obj[node]}\n")
        else:
            with open(data_path + "/" + cost_name, "r") as f:
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    self.costs_obj.append(float(line.rstrip("\n")))


        self.budget = budget

    @property
    def ground_set(self):
        return self.objs

    def objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        S = list(S)
        if len(S) == 0:
            return 0
        S.append(self.feature_count)
        x_s = self.samples.take(S, axis=1)
        p_x_s = pd.DataFrame(x_s)
        counts = p_x_s.value_counts().values/x_s.shape[0]
        ent = entropy(counts)
        return ent

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]


class SensorPlacement(BaseTask):
    def __init__(self, budget: float, n: int = None, data_path: str = None, knapsack=True, seed=0,
                 prepare_max_pair=True, print_curvature=False, construct_graph=False, min_cost=0.4, factor=4.0, cost_mode = "normal"):
        """
        Inputs:
        - n: max_nodes
        - b: budget
        """
        super().__init__()
        if data_path is None:
            raise Exception("Please provide a graph.")
        np.random.seed(seed)
        random.seed(seed)

        self.sensors = []

        # printed = False

        with open(data_path + "/" + "t_data.txt", "r") as f:
            while True:
                line = f.readline()
                if line != "":
                    sensor = list(line.rstrip("\n").split(" "))
                    if "" in sensor:
                        sensor.remove("")
                    self.sensors.append(sensor)
                else:
                    break

        self.sensors = np.array(self.sensors)
        self.sensors.transpose()

        print(self.sensors.shape)

        self.objs = list(range(0, len(self.sensors)))

        self.costs_obj = []

        cost_name = "costs.txt"

        # cost parameters
        if construct_graph:
            if knapsack:
                # self.objs.sort(key=lambda x: len(self.nodes[x]), reverse=True)
                if cost_mode == "normal":
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        (min_cost + random.random()) * factor
                        for node in range(0, len(self.objs))
                    ]
                elif cost_mode == "small":
                    cost_name = "small_costs.txt"
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        max(min_cost, random.gauss(mu=min_cost, sigma=1) * factor)
                        for node in range(0, len(self.objs))
                    ]
                elif cost_mode == "big":
                    cost_name = "big_costs.txt"
                    self.costs_obj = [
                        # self.beta * (len(list(self.graph.neighbors(str(node)))) + 1 - self.alpha)/len(self.nodes)
                        min(min_cost + 1, random.gauss(mu=min_cost + 1, sigma=1) * factor)
                        for node in range(0, len(self.objs))
                    ]
            else:
                # cardinality
                self.costs_obj = [
                    1
                    for node in range(0, len(self.objs))
                ]
            with open(data_path + "/" + cost_name, "w") as f:
                for node in range(0, len(self.objs)):
                    f.write(f"{self.costs_obj[node]}\n")
        else:
            with open(data_path + "/" + cost_name, "r") as f:
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    self.costs_obj.append(float(line.rstrip("\n")))

        self.budget = budget

    @property
    def ground_set(self):
        return self.objs

    def objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        S = list(S)
        if len(S) == 0:
            return 0

        x_s = self.sensors.take(S, axis=1)
        p_x_s = pd.DataFrame(x_s)
        counts = p_x_s.value_counts().values / x_s.shape[0]
        ent = entropy(counts)
        return ent

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]