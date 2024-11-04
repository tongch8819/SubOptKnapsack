import random

import numpy as np

from base_task import BaseTask
from matroid import Matroid
from compute_knapsack_exp import model_factory

import scipy


class Optimizer:
    def __init__(self):
        self.c = None
        self.b = None
        self.S = None
        self.model: BaseTask = None
        self.w = None
        self.diag_s = None

    def knapsack_constraint(self, c, b):
        self.c = c
        self.b = b
        return self

    def cost(self, c):
        self.c = c
        return self

    def budget(self, b):
        self.b = b
        return self

    def set_model(self, model: BaseTask):
        self.model = model
        self.c = np.array(model.costs_obj)
        self.b = model.budget

        self.S = np.identity(len(model.costs_obj))

        self.w = np.array([
            -model.objective([x]) for x in self.model.ground_set
        ])

        return self


    def permutation_max(self):
        diag_s = [0] * len(self.model.ground_set)
        t = self.model.ground_set
        t.sort(key=lambda x: self.model.density(x, []), reverse=True)

        for i in range(0, len(t)):
            ele = t[i]
            diag_s[ele] = self.model.marginal_gain(ele, t[:i])/self.model.objective([ele])

        self.S = np.diag(diag_s)
        self.diag_s = diag_s

    def permutation_random(self, seed):
        diag_s = [0] * len(self.model.ground_set)

        random.seed(seed)
        t = self.model.ground_set
        random.shuffle(t)

        for i in range(0, len(t)):
            ele = t[i]
            diag_s[ele] = self.model.marginal_gain(ele, t[:i])/self.model.objective([ele])

        self.S = np.diag(diag_s)
        self.diag_s = diag_s

    def sequence(self):
        pass

    def matroid_constraint(self, m: Matroid):
        pass

    def optimize(self):
        bounds = [(0, 1) for _ in range(0, len(self.model.ground_set))]

        w_s = np.matmul(self.S, self.w)
        # c_s = np.matmul(self.S, self.c)

        x = scipy.optimize.linprog(c=w_s, A_ub=[self.c], b_ub=[self.b], bounds=bounds).x

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])

        return {
            "upb": -np.matmul(w_s, x),
            "x": fs
        }
