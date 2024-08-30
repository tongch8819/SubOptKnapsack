import random
from abc import ABC, abstractclassmethod
from typing import List, Set
from copy import deepcopy
import numpy as np

from matroid import Matroid


class BaseTask(ABC):

    def __init__(self):
        self.costs_obj = None
        self.max_2_pair = None

        self.max_pair_dict = None
        self.max_pair_array = None

        self.matroid = None

        self.budget = None

        self.c2 = None
        self.d = None
        self.e = None

        self.objective_style = "internal"
        self.objective_dict = {
            "internal": self.internal_objective,
            "mp1_empty": self.mp1_empty_objective,
            "mp1": self.mp1_objective
        }

        self.Y = None
        self.Y_value = 0
        self.empty_Y_value = 0

    def prepare_max_2_pair(self):
        self.max_2_pair = {}

        for i in range(0, len(self.costs_obj)):
            maximal = 0
            for j in range(0, len(self.costs_obj)):
                if i != j and self.objective({i, j}) > maximal:
                    maximal = self.objective({i, j})
            self.max_2_pair[i] = maximal

    def set_modular_obj(self, Y: List[int]):
        pass

    def enable_matroid(self):
        self.matroid = Matroid(self.ground_set, 10, 20)

    def assign_costs(self, knapsack, cost_mode):
        if knapsack:
            # self.objs.sort(key=lambda x: len(self.nodes[x]), reverse=True)
            if cost_mode == "normal":
                self.costs_obj = [
                    (0.4 + random.random()) * 4
                    for obj in self.ground_set
                ]
            elif cost_mode == "integer":
                self.costs_obj = [
                    random.randint(1, 5)
                    for obj in self.ground_set
                ]
        else:
            # cardinality
            self.costs_obj = [
                1
                for node in self.ground_set
            ]
        pass

    @property
    def ground_set(self):
        return None

    @property
    def curvature(self):
        ground_value = self.objective(self.ground_set)

        curvatures = [1-(ground_value-self.objective(set(self.ground_set)-set({x})))/(self.objective(set({x}))) for x in self.ground_set]

        return max(curvatures)

    def print_curvature(self):
        # print(f"start print. curvature:{self.curvature}")

        eles = list(self.ground_set)
        eles.sort(key=lambda x: self.density(x, set()), reverse=True)
        f = self.objective

        c2 = 0.
        d = []

        sum = 0

        for i in range(0, min(100, len(eles))):
            sum += f({eles[i]})
            if eles[i] >= 0:
                prev = 0.
                if i > 0:
                    prev = f(set(eles[:i]))

                prev_sin = 0.
                if i > 0:
                    prev_sin = f({eles[i-1]})

                # print(f"f(A):{f(set(eles[:i+1]))},sum:{sum},sin:{f({eles[i]})},sub:{f(set(eles[:i+1])) - prev},ele:{eles[i]}, decreased:{prev_sin - f({eles[i]})} ")

                c2 += f({eles[i]})/((f(set(eles[:i+1])) - prev) * (i+1))

                if i > 0:
                    d.append(prev_sin - f({eles[i]}))

        d = np.sum(d)/len(d)

        e = c2 * d

        self.c2 = c2
        self.d = d
        self.e = e

        print(f"c:{self.curvature}, c2:{c2}, d:{d}, e:{e}")

    def calculate_m(self):
        ret = [self.cutout_density(ele, self.ground_set) for ele in self.ground_set]
        return ret

    def objective(self, obj: List[int]):
        return self.objective_dict[self.objective_style](obj)

    # @abstractclassmethod
    def internal_objective(self, S: List[int]):
        pass

    def mp1_empty_objective(self, S: List[int]):
        return sum([self.internal_objective([ele]) for ele in S])

    def set_Y(self, Y):
        self.Y = Y
        self.Y_value = self.internal_objective(self.Y)

        term_2 = 0.
        for ele in set(self.Y):
            term_2 += self.internal_cutout_marginal_gain(ele)

        self.empty_Y_value = self.Y_value - term_2

    def mp1_objective(self, S: List[int]):
        term_1 = self.Y_value

        term_2 = 0.
        for ele in set(self.Y) - set(S):
            term_2 += self.internal_cutout_marginal_gain(ele)

        term_3 = 0.
        for ele in set(S) - set(self.Y):
            term_3 += self.internal_marginal_gain(ele, self.Y)

        return term_1 - term_2 + term_3 - self.empty_Y_value

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]

    def marginal_gain(self, single: int, base: List[int]):
        if len(base) == 0:
            base2 = [single]
        else:
            base = list(base)
            base2 = list(set(base + [single]))
        fS1 = self.objective(base)
        fS2 = self.objective(base2)
        res = fS2 - fS1
        assert res >= -0.01, f"f({base2}) - f({base}) = {fS2:.2f} - {fS1:.2f}"
        return res

    def internal_marginal_gain(self, single: int, base: List[int]):
        if len(base) == 0:
            base2 = [single]
        else:
            base = list(base)
            base2 = list(set(base + [single]))
        fS1 = self.internal_objective(base)
        fS2 = self.internal_objective(base2)
        res = fS2 - fS1
        assert res >= -0.01, f"f({base2}) - f({base}) = {fS2:.2f} - {fS1:.2f}"
        return res

    def density(self, single: int, base: List[int]):
        mg = self.marginal_gain(single, base)
        # print(f"s:{single}, c:{self.costs_obj}")
        cost = self.costs_obj[single]
        return (mg * 100) / (cost * 100)

    # def cutout_marginal_gain(self, singleton: int, base: Set[int]):
    #     if type(base) is not set:
    #         base = set(base)
    #     assert singleton in base
    #     # assert type(base) is set, "{} is not set".format(type(base))
    #     base2 = deepcopy(base)
    #     base2.remove(singleton)  # no return value
    #     fS2, fS1 = self.objective(base), self.objective(base2)
    #     res = fS2 - fS1
    #     assert res >= 0., f"f({base}) - f({base2}) = {fS2:.2f} - {fS1:.2f}\n{base - base2}"
    #     return res

    def cutout_marginal_gain(self, singleton: int):
        # assert type(base) is set, "{} is not set".format(type(base))
        base = set(self.ground_set)
        base2 = deepcopy(base)
        base2.remove(singleton)  # no return value
        fS2, fS1 = self.objective(base), self.objective(base2)
        res = fS2 - fS1
        assert res >= 0., f"f({base}) - f({base2}) = {fS2:.2f} - {fS1:.2f}\n{base - base2}"
        return res

    def internal_cutout_marginal_gain(self, singleton: int):
        # assert type(base) is set, "{} is not set".format(type(base))
        base = set(self.ground_set)
        base2 = deepcopy(base)
        base2.remove(singleton)  # no return value
        fS2, fS1 = self.internal_objective(base), self.internal_objective(base2)
        res = fS2 - fS1
        assert res >= 0., f"f({base}) - f({base2}) = {fS2:.2f} - {fS1:.2f}\n{base - base2}"
        return res

    def cutout_density(self, singleton: int, base: Set[int]):
        return self.cutout_marginal_gain(singleton) / self.cost_of_singleton(singleton)

    def internal_cutout_density(self, singleton: int, base: Set[int]):
        return self.internal_cutout_marginal_gain(singleton) / self.cost_of_singleton(singleton)
