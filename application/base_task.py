from application.constraint import knapsack_id

from abc import ABC
from typing import List, Set
from copy import deepcopy


class BaseTask(ABC):
    """
    Model objective instead of constraint
    """

    def __init__(self, constraint=None, is_mono=True):
        self.constraint = constraint
        self.is_mono = is_mono
        self.total_curvature = None

    def objective(self):
        pass

    def marginal_gain(self, single: int, base: List[int]):
        if len(base) == 0:
            base2 = [single]
        else:
            base = list(base)
            base2 = list(set(base + [single]))
        fS1 = self.objective(base)
        fS2 = self.objective(base2)
        res = fS2 - fS1
        if self.is_mono:
            assert res >= 0., f"f({base2}) - f({base}) = {fS2:.2f} - {fS1:.2f}"
        return res

    def density(self, singleton: int, base: List[int]):
        """
        return f(singleton | base) / c(singleton)
        """
        if self.constraint is None:
            raise ValueError("Constraint is not defined.")
        if self.constraint.id != knapsack_id:
            raise ValueError("Constraint is not knapsack.")
        mg = self.marginal_gain(singleton, base)
        cost = self.constraint.cost_of_singleton(singleton)
        return (mg * 100) / (cost * 100)

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
        fS2, fS1 = self.objective(base), self.objective(base2)
        res = fS2 - fS1
        assert res >= 0., f"f({base}) - f({base2}) = {fS2:.2f} - {fS1:.2f}\n{base - base2}"
        return res

    def cutout_density(self, singleton: int, base: Set[int]):
        return self.cutout_marginal_gain(singleton, base) / self.cost_of_singleton(singleton)

    def gen_total_curvature(self):
        if self.total_curvature is not None:
            return self.total_curvature

        V = self.ground_set
        min_ratio = 1
        for e in self.ground_set:
            nume = self.cutout_marginal_gain(e, V)
            denom = self.objective([e])
            assert denom >= nume
            ratio = nume / (denom + 1e-7)
            min_ratio = min(min_ratio, ratio)
        self.total_curvature = 1 - min_ratio
        return self.total_curvature
