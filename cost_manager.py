import random

import numpy as np


class CostManager:
    def __init__(self):
        self.mode = None
        self.model = None
        self.assign = None

        self.minimal_cost = 1.6
        self.maximal_cost = 5.6

        pass

    def set_mode(self, mode):
        self.mode = mode

    def set_model(self, model):
        self.model = model

    def build(self):
        assert self.mode is not None

        if self.mode == 'normal':
            self.assign = self.assign_random
        elif self.mode == 'positive':
            self.assign = self.assign_positive
        elif self.mode == 'negative':
            self.assign = self.assign_negative
        elif self.mode == 'www1':
            self.assign = self.assign_www1
        elif self.mode == 'www2':
            self.assign = self.assign_www2
        else:
            raise Exception(f"Mode {self.mode} does not exist.")

        pass

    def assign_random(self):
        pass

    def assign_positive(self):
        min_idx, min_v = None, 0
        max_idx, max_v = None, 0

        for i in range(0, len(self.model.ground_set)):
            if min_idx is None or min_v > self.model.objective([i]):
                min_idx, min_v = i, self.model.objective([i])

            if max_idx is None or max_v < self.model.objective([i]):
                max_idx, max_v = i, self.model.objective([i])

        factor = (max_v - min_v) / (self.maximal_cost - self.minimal_cost)

        costs = []
        for i in range(0, len(self.model.ground_set)):
            costs.append(self.minimal_cost + (self.model.objective([i]) - min_v) / factor)

        # print(f"c:{costs[:10]}, {[self.model.objective([i]) for i in range(0, 10)]}")

        return costs

    def assign_negative(self):
        min_idx, min_v = None, 0
        max_idx, max_v = None, 0

        for i in range(0, len(self.model.ground_set)):
            if min_idx is None or min_v > self.model.objective([i]):
                min_idx, min_v = i, self.model.objective([i])

            if max_idx is None or max_v < self.model.objective([i]):
                max_idx, max_v = i, self.model.objective([i])

        factor = (max_v - min_v) / (self.maximal_cost - self.minimal_cost)

        costs = []
        for i in range(0, len(self.model.ground_set)):
            costs.append(self.maximal_cost - (self.model.objective([i]) - min_v) / factor)

        # print(f"c:{costs[:10]}, {[self.model.objective([i]) for i in range(0, 10)]}")

        return costs

    def assign_www1(self):
        n = len(self.model.ground_set)
        small_part_n = int(0.8 * n)
        total_idx_pool = list(range(0, n))
        small_idx_pool = random.sample(total_idx_pool, small_part_n)
        large_idx_pool = list(set(total_idx_pool) - set(small_idx_pool))

        costs = [0] * n

        for i in small_idx_pool:
            costs[i] = 1 + 4 * random.random()

        for i in large_idx_pool:
            costs[i] = 5 + 15 * random.random()

        self.model.A = np.matrix([costs])
        self.model.cc = 1

        return costs

    def assign_www2(self):
        n = len(self.model.ground_set)
        small_part_n = int(0.8 * n)
        total_idx_pool = list(range(0, n))
        small_idx_pool = random.sample(total_idx_pool, small_part_n)
        mini_idx_pool = list(set(total_idx_pool) - set(small_idx_pool))

        costs = [0] * n

        for i in small_idx_pool:
            costs[i] = 1 + 4 * random.random()

        for i in mini_idx_pool:
            costs[i] = random.random()

        self.model.A = np.matrix([costs])
        self.model.cc = 1

        return costs
