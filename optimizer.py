import random
import shlex
from typing import Set, List

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

        x = scipy.optimize.linprog(c=w_s, A_ub=[self.c], b_ub=[self.b], bounds=bounds).x

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])

        return {
            "upb": -np.matmul(w_s, x),
            "x": fs
        }


class UpperBoundFunction:
    def __init__(self, f, ground: List[int]):
        self.f = f
        self.ground = list(ground)

        self.Y = None
        self.w = None
        self.genre = ""

        self.base_value = None
        self.etw_dict = {}

    def setF(self, f):
        self.f = f
        return self

    def setGround(self, ground):
        self.ground = list(ground)
        return self

    def setY(self, Y):
        self.Y = set(Y)
        return self

    def setType(self, genre):
        self.genre = genre
        return self

    def oracle(self, S):
        ret = self.base_value
        for ele in S:
            ret += self.w[self.etw_dict[ele]]
        return ret

    def build(self):
        def margin(S, base):
            item1 = set(S) | set(base)
            v1 = self.f(list(item1))
            item2 = set(base)
            v2 = self.f(list(item2))
            return v1 - v2

        for ele_idx in range(0, len(self.ground)):
            ele = self.ground[ele_idx]
            self.etw_dict[ele] = ele_idx

        Y_value = self.f(self.Y)
        self.w = [0] * len(self.ground)

        self.base_value = 0

        if self.genre == 'm1+':
            self.base_value = Y_value
            for v in self.Y:
                self.base_value -= margin({v}, set(self.ground) - {v})

            for ele_idx in range(0, len(self.ground)):
                ele = self.ground[ele_idx]
                item_1 = Y_value

                item_2 = 0
                for v in self.Y - {ele}:
                    item_2 += margin({v}, set(self.ground) - {v})

                item_3 = 0
                for v in {ele} - self.Y:
                    item_3 += margin({v}, self.Y)

                self.w[ele_idx] = item_1 - item_2 + item_3 - self.base_value
        elif self.genre == 'm2+':
            self.base_value = Y_value
            for v in self.Y:
                self.base_value -= margin({v}, self.Y - {v})

            for ele_idx in range(0, len(self.ground)):
                ele = self.ground[ele_idx]
                item_1 = Y_value

                item_2 = 0
                for v in self.Y - {ele}:
                    item_2 += margin({v}, self.Y - {v})

                item_3 = 0
                for v in {ele} - self.Y:
                    item_3 += self.f({v})

                self.w[ele_idx] = item_1 - item_2 + item_3 - self.base_value

class PackingOptimizer:
    def __init__(self):
        self.A = None
        self.b = None
        self.S = None
        self.model: BaseTask = None

        self.permutation_mode = 'max'

        self.w = None
        self.diag_s = None
        self.base = set()
        self.base_value = 0
        self.remaining = None

        self.upb_function = None

    def budget(self, b):
        self.b = b
        return self

    def f_s(self, s):
        return self.model.objective(list(set(s) | self.base)) - self.base_value

    def setBase(self, base):
        self.base = set(base)
        self.base_value = self.model.objective(list(base))
        return self

        # self.remaining = set(self.model.ground_set) - set(self.base)
        # self.A = self.model.A[:, list(self.remaining)]
        #
        # self.S = np.identity(len(self.remaining))
        # self.diag_s = [0] * len(self.remaining)
        #
        # self.w = np.array([
        #     -self.f_s([x]) for x in self.remaining
        # ])

    def setModel(self, model: BaseTask):
        self.model = model
        self.remaining = self.model.ground_set
        self.A = model.A
        self.b = model.bv

        self.S = np.identity(len(self.remaining))

        self.w = np.array([
            -model.objective([x]) for x in self.model.ground_set
        ])
        return self

    def sample(self, n):
        return random.sample(self.remaining, n)

    def permutation_max(self):
        diag_s = [0] * len(self.remaining)
        t = self.remaining
        t.sort(key=lambda x: self.model.density(x, list(self.base)), reverse=True)

        for i in range(0, len(t)):
            ele = t[i]
            diag_s[ele] = self.model.marginal_gain(ele, list(set(t[:i]) | self.base))/self.f_s([ele])

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

    def build(self):
        # update base
        self.remaining = self.model.ground_set
        self.b = self.model.bv

        self.base_value = self.model.objective(list(self.base))
        self.remaining = set(self.model.ground_set) - set(self.base)
        self.A = self.model.A[:, list(self.remaining)]

        if self.permutation_mode == 'none':
            self.S = np.identity(len(self.remaining))
            self.diag_s = [0] * len(self.remaining)
        elif self.permutation_mode == 'max':
            # update permutation
            diag_s = [0] * len(self.remaining)
            t = list(self.remaining)
            t.sort(key=lambda x: self.model.density(x, list(self.base)), reverse=True)

            for i in range(0, len(t)):
                ele = t[i]
                diag_s[ele] = self.model.marginal_gain(ele, list(set(t[:i]) | self.base))/self.f_s([ele])

            self.S = np.diag(diag_s)
            self.diag_s = diag_s
        if self.upb_function == None:
            self.w = np.array([
                -self.f_s([x]) for x in self.remaining
            ])
        else:
            self.upb_function.setY(set(self.upb_function.Y) & set(self.remaining))
            self.upb_function.setF(self.f_s)
            self.upb_function.setGround(self.remaining)
            self.upb_function.build()

            self.w = -np.array(self.upb_function.w)


    def optimize(self):
        bounds = [(0, 1) for _ in range(0, len(self.remaining))]
        # print(f"2 S:{self.S.shape}, A:{self.A.shape}")
        w_s = np.matmul(self.S, self.w)
        A_s = np.matmul(self.A, self.S)

        upb_base = 0.
        if self.upb_function is not None:
            upb_base = self.upb_function.base_value

        x = scipy.optimize.linprog(c=w_s, A_ub=A_s, b_ub=self.b, bounds=bounds).x

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])

        return {
            "upb": -np.matmul(w_s, x) + self.base_value + upb_base,
            "x": fs
        }
