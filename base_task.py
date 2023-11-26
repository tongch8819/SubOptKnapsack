from abc import ABC, abstractclassmethod
from typing import List, Set
from copy import deepcopy


class BaseTask(ABC):

    @property
    def budget(self):
        return self.b

    @abstractclassmethod
    def objective(self):
        pass

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
        assert res >= 0., f"f({base2}) - f({base}) = {fS2:.2f} - {fS1:.2f}"
        return res

    def density(self, single: int, base: List[int]):
        mg = self.marginal_gain(single, base)
        cost = self.costs_obj[single]
        return (mg * 100) / (cost * 100)

    def cutout_marginal_gain(self, singleton: int, base: Set[int]):
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
