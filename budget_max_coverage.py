from base_task import BaseTask
import random
from collections import defaultdict
from typing import List, Set
from copy import deepcopy



class IdealMaxCovModel(BaseTask):
    def __init__(self, n1=100, n2=100, p=0.02, budget=1):
        # number of words
        self.n1 = n1
        # number of objects
        self.n2 = n2
        # relation probability
        self.p = p
        random.seed(1)

        words = [i for i in range(n1)]
        self.objects = [i for i in range(n2)]  # immutable tuple
        edges = []
        self.obj2words = defaultdict(set)

        for obj in self.objects:
            for wrd in words:
                if random.random() <= p:
                    edges.append((obj, wrd))
                    self.obj2words[obj].add(wrd)

        # cost function: object -> real value
        self.costs_obj = [random.random() for i in range(n2)]
        self.b = budget

    @property
    def ground_set(self):
        return self.objects

    @property
    def budget(self):
        return self.b

    def objective(self, S: List[int]):
        """submodular objective"""
        res = set()
        S = list(S)
        for obj in S:
            res = res.union(self.obj2words[obj])
        return len(res)

    # def marginal_gain(self, single: int, base: List[int]):
    #     if len(base) == 0:
    #         base2 = [single]
    #     else:
    #         base2 = list(set(list(base) + [single]))
    #     fS1 = self.objective(base)
    #     fS2 = self.objective(base2)
    #     return fS2 - fS1

    # def density(self, single: int, base: List[int]):
    #     mg = self.marginal_gain(single, base)
    #     cost = self.costs_obj[single]
    #     return (mg * 100100) / (cost * 100100)

    # def cost_of_set(self, S: List[int]):
    #     return sum(self.costs_obj[x] for x in S)
    
    # def cost_of_singleton(self, singleton: int):
    #     assert singleton < len(self.costs_obj), "Singleton: {}".format(singleton)
    #     return self.costs_obj[singleton]

    # def cutout_marginal_gain(self, singleton: int, base: Set):
    #     assert singleton in base
    #     assert type(base) is set
    #     base2 = deepcopy(base)
    #     base2.remove(singleton)  # no return value
    #     return self.objective(base) - self.objective(base2)
    
    # def cutout_density(self, singleton: int, base: Set):
    #     return self.cutout_marginal_gain(singleton, base) / self.cost_of_singleton(singleton)