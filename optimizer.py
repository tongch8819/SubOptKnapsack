import copy
import math
import random
import shlex
from typing import Set, List

import numpy as np

from base_task import BaseTask
from matroid import Matroid

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
            diag_s[ele] = self.model.marginal_gain(ele, t[:i]) / self.model.objective([ele])

        self.S = np.diag(diag_s)
        self.diag_s = diag_s

    def permutation_random(self, seed):
        diag_s = [0] * len(self.model.ground_set)

        random.seed(seed)
        t = self.model.ground_set
        random.shuffle(t)

        for i in range(0, len(t)):
            ele = t[i]
            diag_s[ele] = self.model.marginal_gain(ele, t[:i]) / self.model.objective([ele])

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
            diag_s[ele] = self.model.marginal_gain(ele, list(set(t[:i]) | self.base)) / self.f_s([ele])

        self.S = np.diag(diag_s)
        self.diag_s = diag_s

    def permutation_random(self, seed):
        diag_s = [0] * len(self.model.ground_set)

        random.seed(seed)
        t = self.model.ground_set
        random.shuffle(t)

        for i in range(0, len(t)):
            ele = t[i]
            diag_s[ele] = self.model.marginal_gain(ele, t[:i]) / self.model.objective([ele])

        self.S = np.diag(diag_s)
        self.diag_s = diag_s

    def build(self):
        # update base
        self.remaining = self.model.ground_set
        self.b = np.array(self.model.bv, dtype=float)
        # for c_idx in range(0, self.b.shape[0]):
        #     for ele in self.base:
        #         self.b[c_idx] -= self.model.A[c_idx, ele]

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
                diag_s[ele] = self.model.marginal_gain(ele, list(set(t[:i]) | self.base)) / self.f_s([ele])

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
        valid = True
        for b_i in self.b:
            if b_i < 0:
                valid = False
        if not valid:
            return {
                "upb": 10000,
                "x": np.zeros(len(self.remaining))
            }

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

        # print(f"v1:{-np.matmul(w_s, x)}")
        # print(f"v2:{self.base_value}")

        return {
            "upb": -np.matmul(w_s, x) + self.base_value + upb_base,
            "x": fs
        }


class PackingModified1Optimizer:
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
            diag_s[ele] = self.model.marginal_gain(ele, list(set(t[:i]) | self.base)) / self.f_s([ele])

        self.S = np.diag(diag_s)
        self.diag_s = diag_s

    def permutation_random(self, seed):
        diag_s = [0] * len(self.model.ground_set)

        random.seed(seed)
        t = self.model.ground_set
        random.shuffle(t)

        for i in range(0, len(t)):
            ele = t[i]
            diag_s[ele] = self.model.marginal_gain(ele, t[:i]) / self.model.objective([ele])

        self.S = np.diag(diag_s)
        self.diag_s = diag_s

    def build(self):
        # update base
        self.remaining = self.model.ground_set
        self.b = np.array(self.model.bv, dtype=float)
        for c_idx in range(0, self.b.shape[0]):
            for ele in self.base:
                self.b[c_idx] -= self.model.A[c_idx, ele]

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
                diag_s[ele] = self.model.marginal_gain(ele, list(set(t[:i]) | self.base)) / self.f_s([ele])

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
        valid = True
        for b_i in self.b:
            if b_i < 0:
                valid = False
        if not valid:
            return {
                "upb": 10000,
                "x": np.zeros(len(self.remaining))
            }

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

        # print(f"v1:{-np.matmul(w_s, x)}")
        # print(f"v2:{self.base_value}")

        return {
            "upb": -np.matmul(w_s, x) + self.base_value + upb_base,
            "x": fs
        }


class PackingModifiedOptimizer:
    def __init__(self):
        self.A = None
        self.bv = None

        self.model: BaseTask = None

        self.w = None
        self.base = set()
        self.base_value = 0
        self.remaining = None

        self.d = None
        # f(A_i)
        self.fA = None
        self.permutation_mode = None

    def budget(self, bv):
        self.bv = bv
        return self

    def f_s(self, s):
        return self.model.objective(list(set(s) | self.base)) - self.base_value

    def setBase(self, base):
        self.base = set(base)
        self.base_value = self.model.objective(list(base))
        return self

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

    def build(self):
        # update base
        self.remaining = list(set(self.model.ground_set) - set(self.base))
        self.bv = self.model.bv
        # update d and fA
        m = len(self.bv)
        n = len(self.remaining)
        self.d = np.zeros(shape=(m, n))

        for c_idx in range(0, m):
            for e_idx in range(0, n):
                self.d[c_idx, e_idx] = self.model.density_A(self.remaining[e_idx], list(self.base), c_idx)

        self.fA = np.zeros(n)
        for e_idx in range(0, n):
            self.fA[e_idx] = self.f_s(self.remaining[:e_idx + 1])

        self.base_value = self.model.objective(list(self.base))
        self.remaining = list(set(self.model.ground_set) - set(self.base))
        self.A = self.model.A[:, list(self.remaining)]

    def optimize(self):
        # print(f"2 S:{self.S.shape}, A:{self.A.shape}")
        m = len(self.bv)
        n = len(self.remaining)
        bounds = []
        for i in range(0, n):
            bounds.append((0, 100))

        cost_bound = np.zeros(m * n)
        for c_idx in range(0, m):
            for e_idx in range(0, n):
                cost_bound[e_idx + c_idx * n] = self.model.A[c_idx, self.remaining[e_idx]]

        for i in range(n, n + m * n):
            bounds.append((0, cost_bound[i - n]))

        bounds.append((1, 1))
        # vector d
        # vector A
        # vector b

        # the constraint matrix should be in shape (mn + n) * (n + m * n + 1)
        A = np.zeros(shape=(m * n + n + m, n + m * n + 1))
        b = np.zeros(shape=(m * n + n + m, 1))
        # the vector to optimize is in the form of (v_1, v_2, ..., v_n, s^1_1, s^1_2, ..., s^1_n, ..., s^m_1, s^m_2, ..., s^m_n 1)

        # the additive bit
        additive_idx = n + m * n

        # condition (ii)
        # this type of constraints occupies the first m rows
        for c_idx in range(0, m):
            for e_idx in range(0, n):
                A[c_idx, e_idx + (c_idx + 1) * n] = 1
                A[c_idx, additive_idx] = -self.model.bv[c_idx]

        # condition (iii)
        # this type of constraints occupies the following m * n rows

        # outer loop: constraint index
        # inner loop: element index
        for c_idx in range(0, m):
            for e_idx in range(0, n):
                # row of this constraint in A
                r = c_idx * n + e_idx + m
                # v_i
                A[r, e_idx] = 1
                # s_i
                # print(f"A:{type(A)}, d:{type(self.d)}")
                A[r, e_idx + (c_idx + 1) * n] = -self.d[c_idx][e_idx]

        # condition (iv)
        # this type of constraints occupies the following n rows

        start_row = m * n + m
        for c_iii_offset in range(0, n):
            r = start_row + c_iii_offset
            for v_idx in range(0, c_iii_offset + 1):
                A[r, v_idx] = 1
            A[r][additive_idx] = -self.fA[c_iii_offset]
        # print(f"A:{A}")

        # just accumulate all v_i
        w = np.zeros(n + m * n + 1)
        for i in range(0, n):
            w[i] = -1

        # upb_base = 0.
        # if self.upb_function is not None:
        #     upb_base = self.upb_function.base_value
        # print(f"A:{A.shape}, b:{b.shape},bounds:{len(bounds)}")
        x = scipy.optimize.linprog(c=w, A_ub=A, b_ub=b, bounds=bounds).x

        # print(f"x:{x}")
        # print(f"d:{self.d}")
        # print(f"A:{self.fA}")
        # # print(f"bv:{self.model.bv}")
        # # print(f"total budget:{np.sum(x[n: 2*n])}")
        # print(f"r:{self.remaining}")
        # print(f"base:{self.base}, base v:{self.base_value}")
        # print(f"l1:{-np.matmul(w, x)}")
        # print(f"l2:{-np.matmul(w, x)+ self.base_value}")

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])

        return {
            "upb": -np.matmul(w, x) + self.base_value,
            "x": fs
        }

    # def optimize(self):
    #     # print(f"2 S:{self.S.shape}, A:{self.A.shape}")
    #
    #     m = len(self.bv)
    #     n = len(self.remaining)
    #     bounds = []
    #     for i in range(0, n):
    #         bounds.append((0, 100))
    #
    #     cost_bound = np.zeros(n)
    #     for e_idx in range(0, n):
    #         max_cost = self.model.A[0, e_idx]
    #         for c_idx in range(1, m):
    #             # print(f"a:{self.model.A.shape}, c:{c_idx}, e:{e_idx}")
    #             if self.model.A[c_idx, e_idx] > max_cost:
    #                 max_cost = self.model.A[c_idx, e_idx]
    #         cost_bound[e_idx] = max_cost
    #
    #     for i in range(n, 2 * n):
    #         bounds.append((0, cost_bound[i - n]))
    #     bounds.append((1, 1))
    #     # vector d
    #     # vector A
    #     # vector b
    #
    #     # the constraint matrix should be in shape (mn + n) * (2n + 1)
    #     A = np.zeros(shape=(m * n + n + m, 2 * n + 1))
    #     b = np.zeros(shape=(m * n + n + m, 1))
    #     # the vector to optimize is in the form of (v_1, v_2, ..., v_n, s_1, s_2, ..., s_n, 1)
    #
    #     # the additive bit
    #     additive_idx = 2 * n
    #
    #     # condition (ii)
    #     # this type of constraints occupies the first m rows
    #     for c_idx in range(0, m):
    #         for e_idx in range(0, n):
    #             A[c_idx, e_idx + n] = 1
    #             A[c_idx, additive_idx] = -self.model.bv[c_idx]
    #
    #     # condition (iii)
    #     # this type of constraints occupies the following m * n rows
    #
    #     # outer loop: constraint index
    #     # inner loop: element index
    #     for c_idx in range(0, m):
    #         for e_idx in range(0, n):
    #             # row of this constraint in A
    #             r = c_idx * n + e_idx + m
    #             # v_i
    #             A[r][e_idx] = 1
    #             # s_i
    #             # print(f"A:{type(A)}, d:{type(self.d)}")
    #             A[r][e_idx + n] = -self.d[c_idx][e_idx]
    #
    #     # condition (iv)
    #     # this type of constraints occupies the following n rows
    #
    #     start_row = m * n + m
    #     for c_iii_offset in range(0, n):
    #         r = start_row + c_iii_offset
    #         for v_idx in range(0, c_iii_offset + 1):
    #             A[r][v_idx] = 1
    #         A[r][additive_idx] = -self.fA[c_iii_offset]
    #     # print(f"A:{A}")
    #
    #     # just accumulate all v_i
    #     w = np.zeros(2 * n + 1)
    #     for i in range(0, n):
    #         w[i] = -1
    #
    #     # upb_base = 0.
    #     # if self.upb_function is not None:
    #     #     upb_base = self.upb_function.base_value
    #
    #     x = scipy.optimize.linprog(c=w, A_ub=A, b_ub=b, bounds=bounds).x
    #
    #     print(f"x:{x}")
    #     print(f"d:{self.d}")
    #     print(f"A:{self.fA}")
    #     print(f"bv:{self.model.bv}")
    #     print(f"total budget:{np.sum(x[n: 2*n])}")
    #     print(f"base:{self.base_value}")
    #     print(f"l1:{-np.matmul(w, x)}")
    #     print(f"l2:{-np.matmul(w, x)+ self.base_value}")
    #     fs = {}
    #     for i in range(0, len(x)):
    #         if x[i] > 0:
    #             fs[i] = float(x[i])
    #
    #     return {
    #         "upb": -np.matmul(w, x) + self.base_value,
    #         "x": fs
    #     }


class PackingModified2Optimizer:
    def __init__(self):
        self.A = None
        self.A_plus = None
        self.bv = None

        self.model: BaseTask = None

        self.w = None
        self.base = set()
        self.base_value = 0
        self.remaining = None

        # f(A_i)
        self.fA = None
        self.permutation_mode = None

    def budget(self, bv):
        self.bv = bv
        return self

    def f_s(self, s):
        if type(s) is int:
            s = {s}
        return self.model.objective(list(set(s) | self.base)) - self.base_value

    def setBase(self, base):
        self.base = set(base)
        self.base_value = self.model.objective(list(base))
        return self

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

    def temp_density(self, x):
        margin = self.f_s({x})
        cost = self.model.A[0, x]
        return (margin * 100) / (cost * 100)

    def build(self):
        # update base
        self.remaining = list(set(self.model.ground_set) - set(self.base))
        self.bv = self.model.bv
        n = len(self.remaining)

        self.base_value = self.model.objective(list(self.base))
        self.remaining = list(set(self.model.ground_set) - set(self.base))
        # self.remaining.sort(key=self.f_s, reverse=True)

        self.fA = np.zeros(n)
        for e_idx in range(0, n):
            self.fA[e_idx] = self.f_s(self.remaining[:e_idx + 1])

        self.A = self.model.A[:, list(self.remaining)]

    def optimize(self):
        # print(f"2 S:{self.S.shape}, A:{self.A.shape}")
        m = len(self.bv)
        n = len(self.remaining)
        bounds = []
        for i in range(0, n):
            bounds.append((0, 100))

        for i in range(0, n):
            bounds.append((0, 1))

        bounds.append((1, 1))
        # vector d
        # vector A
        # vector b

        # the constraint matrix should be in shape (mn + n) * (n + m * n + 1)
        A = np.zeros(shape=(m + 2 * n, 2 * n + 1))
        b = np.zeros(shape=(m + 2 * n, 1))
        # the vector to optimize is in the form of (v_1, v_2, ..., v_n, s^1_1, s^1_2, ..., s^1_n, ..., s^m_1, s^m_2, ..., s^m_n 1)

        # the additive bit
        additive_idx = 2 * n

        # condition (ii)
        # this type of constraints occupies the first m rows
        for c_idx in range(0, m):
            for e_idx in range(0, n):
                # print(f"c:{c_idx}, e:{e_idx}, A:{self.A[c_idx, e_idx]}")
                A[c_idx, e_idx + n] = self.A[c_idx, e_idx]
                A[c_idx, additive_idx] = -self.model.bv[c_idx]

        # condition (iii)
        # this type of constraints occupies the following n rows

        for e_idx in range(0, n):
            # row of this constraint in A
            r = e_idx + m
            # v_i
            A[r, e_idx] = 1
            # s_i
            A[r, e_idx + n] = -self.f_s({self.remaining[e_idx]})

        # condition (iv)
        # this type of constraints occupies the following n rows

        start_row = m + n
        for c_iii_offset in range(0, n):
            r = start_row + c_iii_offset
            for v_idx in range(0, c_iii_offset + 1):
                A[r, v_idx] = 1
            A[r][additive_idx] = -self.fA[c_iii_offset]

        # just accumulate all v_i
        w = np.zeros(2 * n + 1)
        for i in range(0, n):
            w[i] = -1

        x = scipy.optimize.linprog(c=w, A_ub=A, b_ub=b, bounds=bounds).x

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])

        return {
            "upb": -np.matmul(w, x) + self.base_value,
            "x": fs
        }

        # print(f"x:{x}")
        # print(f"d:{self.d}")
        # print(f"A:{self.fA}")
        # # print(f"bv:{self.model.bv}")
        # # print(f"total budget:{np.sum(x[n: 2*n])}")
        # print(f"r:{self.remaining}")
        # print(f"base:{self.base}, base v:{self.base_value}")
        # print(f"l1:{-np.matmul(w, x)+ self.base_value}")

        # second phase
        # vti_dict = {}
        # remaining_plus = self.remaining
        # for e_idx in range(0, n):
        #     vti_dict[remaining_plus[e_idx]] = e_idx
        #
        # remaining_plus = list(set(self.model.ground_set) - set(self.base))
        # remaining_plus.sort(key=lambda ele: x[vti_dict[ele]], reverse=True)
        # # print(f"r:{self.remaining[:10]}, r1:{remaining_plus[:10]}")
        #
        # # print(f"re:{remaining_plus}, v:{[x[vti_dict[ele]] for ele in remaining_plus]}")
        #
        # fA_plus = np.zeros(n)
        # for e_idx in range(0, n):
        #     fA_plus[e_idx] = self.f_s(remaining_plus[:e_idx + 1])
        #
        # self.A_plus = self.model.A[:, list(remaining_plus)]
        #
        # # vector d
        # # vector A
        # # vector b
        #
        # # the constraint matrix should be in shape (mn + n) * (n + m * n + 1)
        # A = np.zeros(shape=(m + 2 * n, 2 * n + 1))
        # b = np.zeros(shape=(m + 2 * n, 1))
        # # the vector to optimize is in the form of (v_1, v_2, ..., v_n, s^1_1, s^1_2, ..., s^1_n, ..., s^m_1, s^m_2, ..., s^m_n 1)
        #
        # # the additive bit
        # additive_idx = 2 * n
        #
        # # condition (ii)
        # # this type of constraints occupies the first m rows
        # for c_idx in range(0, m):
        #     for e_idx in range(0, n):
        #         # print(f"c:{c_idx}, e:{e_idx}, A:{self.A[c_idx, e_idx]}")
        #         A[c_idx, e_idx + n] = self.A_plus[c_idx, e_idx]
        #         A[c_idx, additive_idx] = -self.model.bv[c_idx]
        #
        # # condition (iii)
        # # this type of constraints occupies the following n rows
        #
        # for e_idx in range(0, n):
        #     # row of this constraint in A
        #     r = e_idx + m
        #     # v_i
        #     A[r, e_idx] = 1
        #     # s_i
        #     A[r, e_idx + n] = -self.f_s({remaining_plus[e_idx]})
        #
        # # condition (iv)
        # # this type of constraints occupies the following n rows
        #
        # start_row = m + n
        # for c_iii_offset in range(0, n):
        #     r = start_row + c_iii_offset
        #     for v_idx in range(0, c_iii_offset + 1):
        #         A[r, v_idx] = 1
        #     A[r][additive_idx] = -fA_plus[c_iii_offset]
        # # print(f"A:{A}")
        #
        # x = scipy.optimize.linprog(c=w, A_ub=A, b_ub=b, bounds=bounds).x
        #
        # # print(f"x plus:{x}")
        # # print(f"fA plus:{fA_plus}")
        # # print(f"single plus:{[self.f_s(ele) for ele in remaining_plus]}")
        # # print(f"l2:{-np.matmul(w, x)+ self.base_value}")
        #
        # fs = {}
        # for i in range(0, len(x)):
        #     if x[i] > 0:
        #         fs[i] = float(x[i])
        #
        # return {
        #     "upb": -np.matmul(w, x) + self.base_value,
        #     "x": fs
        # }


class MultilinearOptimizer:
    def __init__(self):
        self.eps = 0.1
        self.alpha = 0.8
        self.sample_count = 0

        self.model = None
        self.n = 0

        # start point
        self.a = None
        self.base = None
        pass

    def F(self, x):
        total_value = 0
        for _ in range(0, self.sample_count):
            s = []
            for i in range(0, self.n):
                if random.random() < x[i]:
                    s.append(i)

            value = self.model.objective(s)
            total_value = total_value + value
        return total_value / self.sample_count

    def partial_derivative(self, x, i):
        total_value = 0
        for _ in range(0, self.sample_count):
            ground = list(set(self.model.ground_set) - {i})
            s = []
            for j in ground:
                if random.random() < x[j]:
                    s.append(j)

            value = self.model.objective(list(set(s) | {i})) - self.model.objective(s)
            # for j in s:
            #     value = value * x[j]
            # for j in set(ground) - set(s):
            #     value = value * (1 - x[j])
            total_value = total_value + value
        return total_value / self.sample_count

    def gradient(self, x):
        g = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(0, n):
            g[i] = self.partial_derivative(x=x, i=i)
        return g

    def evaluate_sample_count(self):
        nominator = 4 * math.log(1.0 / (1 - self.alpha), math.e)
        denominator = math.pow(self.eps, 2)
        return math.ceil(nominator / denominator)

    def setModel(self, model):
        self.model = model
        self.n = len(model.ground_set)
        self.a = np.zeros(self.n)
        self.base = []

    def setBase(self, base):
        self.a = np.zeros(self.n)
        for i in base:
            self.a[i] = 1.0
        self.base = base

    def build(self):
        self.sample_count = 100  # self.evaluate_sample_count()

    def optimize(self):
        # here we optimize x - a rather than x
        # thus the constraint should be A(x-a) <= b - Aa

        # build w
        # print(f"ground:{self.F(self.a)}, f:{self.model.objective(self.base)}, base:{self.base}")
        w = self.gradient(self.a)
        for i in range(0, self.n):
            w[i] = -w[i]

        bounds = np.array([(0, 1)] * self.n)

        # build A
        A = self.model.A

        # build b
        item1 = np.array(self.model.bv)
        item2 = A @ self.a
        b = item1 - item2

        # print(f"b:{b}")
        for i in range(0, b.shape[1]):
            if b[0, i] < 0:
                return {
                    "upb": 10000,
                    "x": {}
                }

        x = scipy.optimize.linprog(c=w, A_ub=A, b_ub=b, bounds=bounds).x

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])
                # print(f"i:{i}, xi:{x[i]}")
        base = self.F(self.a)

        # print(f"item1:{item1}, item2:{item2}, b:{b}, v1:{-np.matmul(w, x)}, v2:{base}, total:{-np.matmul(w, x) + base}")
        return {
            "delta": -np.matmul(w, x),
            "upb": -np.matmul(w, x) + base,
            "x": fs
        }
        pass


class MultilinearOptimizer2:
    def __init__(self):
        self.eps = 0.1
        self.alpha = 0.8
        self.sample_count = 0

        self.model = None
        self.n = 0

        # start point
        self.a = None
        self.base = None
        self.w = None

        # the linear constraint
        self.L_c = None
        self.A = None
        self.b = None

        # the submodular constraint
        self.remaining = None
        self.NL_sub_c = None
        self.base_value = 0
        self.err = 0.01
        pass

    def F(self, x):
        total_value = 0
        for _ in range(0, self.sample_count):
            s = []
            for i in range(0, self.n):
                if random.random() < x[i]:
                    s.append(i)

            value = self.model.objective(s)
            total_value = total_value + value
        return total_value / self.sample_count

    def partial_derivative(self, x, i):
        total_value = 0
        for _ in range(0, self.sample_count):
            ground = list(set(self.model.ground_set) - {i})
            s = []
            for j in ground:
                if random.random() < x[j]:
                    s.append(j)

            value = self.model.objective(list(set(s) | {i})) - self.model.objective(s)
            # for j in s:
            #     value = value * x[j]
            # for j in set(ground) - set(s):
            #     value = value * (1 - x[j])
            total_value = total_value + value
        return total_value / self.sample_count

    def gradient(self, x):
        g = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(0, n):
            g[i] = self.partial_derivative(x=x, i=i)
        return g

    def evaluate_sample_count(self):
        nominator = 4 * math.log(1.0 / (1 - self.alpha), math.e)
        denominator = math.pow(self.eps, 2)
        return math.ceil(nominator / denominator)

    def setModel(self, model):
        self.model = model
        self.n = len(model.ground_set)
        self.a = np.zeros(self.n)
        self.base = []

    def setBase(self, base):
        self.a = np.zeros(self.n)
        for i in base:
            self.a[i] = 1.0
        self.base = base

    def sub_constraint(self, x):
        ret = np.zeros(self.n)

        for i in range(0, self.n):
            y = np.zeros(self.n)
            g_base_set = set()
            for j in range(0, i + 1):
                y[j] = x[j]
                if y[j] > 0:
                    g_base_set.add(j)

            item1 = self.w @ y
            item2 = self.model.objective(list(set(self.base) | g_base_set)) - self.base_value

            ret[i] = item1 + item2
        return ret

    def sub_constraint2(self, x):
        ret = np.zeros(self.n)

        for i in range(0, self.n):
            y = np.zeros(self.n)
            for j in range(0, i + 1):
                y[j] = x[j]

            item1 = self.w @ y
            item2 = 0
            prev = self.base_value
            g_base_set = set(self.base)
            for j in range(0, i + 1):
                if y[j] > 0:
                    g_base_set = g_base_set | {j}
                    item2 = item2 + (self.model.objective(list(g_base_set)) - prev) * y[j]
                    prev = self.model.objective(list(g_base_set))
            # print(f"item1:{item1}, item2:{item2}")
            ret[i] = item1 + item2
        return ret

    def sub_constraint_i(self, x, i):
        g_base_set = set()

        y = np.zeros(self.n)
        for j in range(0, i + 1):
            y[j] = x[j]
            if x[j] > 0:
                g_base_set.add(j)

        item1 = self.w @ y
        item2 = self.model.objective(list(set(self.base) | g_base_set)) - self.base_value

        # if item1 + item2 + self.err < 0:
        #     print(f"i:{i},item1:{item1},2:{item2}, g_bas:{g_base_set}, base:{self.base}, "
        #           f"bv:{self.base_value}, 2:{list(set(self.base) | g_base_set)},"
        #           f"3:{self.model.objective(list(set(self.base) | g_base_set))}, "
        #           f"y:{y},w:{self.w}")

        ret = item1 + item2 + self.err

        return ret

    def sub_constraint_all(self, x):
        g_base_set = set()
        for i in range(0, self.n):
            if x[i] > 0:
                g_base_set.add(i)

        item1 = self.F(x)
        item2 = self.model.objective(list(set(self.base) | g_base_set)) - self.base_value

        ret = item2 - item1

        return ret

    def linear_constraint_i(self, x, i):
        item1 = (self.A[i] @ x)[0, 0]
        item2 = self.b[0, i]
        return item2 - item1

    def build(self):
        self.sample_count = 100
        self.remaining = list(set(self.model.ground_set) - set(self.base))
        self.base_value = self.model.objective(self.base)
        # sub_c = np.zeros(self.n)
        #
        # for ele in range(0, self.n):
        #     sub_c[ele] = self.model.marginal_gain(ele, list(set(self.base) | set(range(0, ele))))

        # build A
        self.A = self.model.A
        item1 = np.array(self.model.bv)
        item2 = self.A @ self.a
        self.b = np.asarray(item1 - item2).flatten()
        # self.L_c = scipy.optimize.LinearConstraint(A=A, lb=np.zeros(shape=(4, 1)), ub=b, keep_feasible=np.zeros(shape=(4, 1)))
        # print(f"shape:{self.A.shape}, a:{self.a.shape}. c:{(self.A @ self.a).shape}, b:{self.b.shape}")
        # print(f"b:{self.b}")
        # print(f"1:{self.A[1]}, a:{self.A}")
        # self.L_c = [
        #     {
        #         'type': 'ineq',
        #         'fun': lambda x: self.linear_constraint_i(x, i),
        #     }
        #     for i in range(0, self.A.shape[0])
        # ]
        # print(f"0:{self.A.shape}, 2:{self.b.shape},3:{type(self.b)}")
        self.L_c = [
            scipy.optimize.LinearConstraint(A=self.A, lb=np.zeros(self.b.shape[0]), ub=self.b)
        ]
        # self.NL_sub_c = [
        #     {
        #         'type': 'ineq',
        #         'fun': lambda x: self.sub_constraint_i(x, i),
        #     }
        #     for i in range(0, self.A.shape[0])
        # ]
        self.NL_sub_c = [
            scipy.optimize.NonlinearConstraint(fun=self.sub_constraint, lb=np.zeros(self.n), ub=np.inf)
        ]

        # build submodualr condition
        # err = 0.1

        # for i in range(0, self.n):
        #     b[i] = self.model.objective(list(set(self.base) | set(range(0, i + 1)))) - base_value + err

        # self.NL_sub_c = scipy.optimize.NonlinearConstraint(fun=self.sub_constraint, lb=0, ub=b)

    def optimize(self):
        # here we optimize x - a rather than x
        # thus the constraint should be A(x-a) <= b - Aa

        # build w
        # print(f"ground:{self.F(self.a)}, f:{self.model.objective(self.base)}, base:{self.base}")
        # w = self.gradient(self.a)
        # for i in range(0, self.n):
        #     w[i] = -w[i]
        self.w = self.gradient(self.a)
        for i in range(0, self.n):
            self.w[i] = -self.w[i]

        # bounds = np.array([(0, 1)] * self.n)
        bounds = [(0, 1)] * self.n
        for i in self.base:
            bounds[i] = (0, 0)

        # build b
        # print(f"b:{b}")
        for i in range(0, self.b.shape[0]):
            if self.b[i] < 0:
                return {
                    "upb": 10000,
                    "x": {}
                }

        # print(f"start optimize")
        x = scipy.optimize.minimize(lambda y: self.w @ y, x0=np.zeros(self.n), constraints=self.L_c + self.NL_sub_c,
                                    bounds=bounds).x

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])
                # print(f"i:{i}, xi:{x[i]}")
        base = self.F(self.a)
        # print(f"lc:{self.L_c}")
        # print(f"v1:{-(self.w @ x)}, base:{base}, w:{self.w}, x:{x}, bi:{self.b[0]}")
        return {
            "delta": -(self.w @ x),
            "upb": -(self.w @ x) + base,
            "x": fs
        }
        pass


class MatroidOptimizer:
    def __init__(self):
        self.eps = 0.1
        self.alpha = 0.8
        self.sample_count = 0

        self.model = None
        self.n = 0
        self.m = 0

        # start point
        self.a = None
        self.base = None
        self.B = None
        self.nabla = None

        self.final_weight = None

        #
        self.remaining = []
        self.base_value = 0
        self.addition = 0

        pass

    def F(self, x):
        total_value = 0
        for _ in range(0, self.sample_count):
            s = []
            for i in range(0, self.n):
                if random.random() < x[i]:
                    s.append(i)

            value = self.model.objective(s)
            total_value = total_value + value
        return total_value / self.sample_count

    def partial_derivative(self, x, i):
        total_value = 0
        for _ in range(0, self.sample_count):
            ground = list(set(self.model.ground_set) - {i})
            s = []
            for j in ground:
                if random.random() < x[j]:
                    s.append(j)

            value = self.model.objective(list(set(s) | {i})) - self.model.objective(s)
            total_value = total_value + value
        return total_value / self.sample_count

    def gradient(self, x):
        g = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(0, n):
            g[i] = self.partial_derivative(x=x, i=i)
        return g

    def evaluate_sample_count(self):
        nominator = 4 * math.log(1.0 / (1 - self.alpha), math.e)
        denominator = math.pow(self.eps, 2)
        return math.ceil(nominator / denominator)

    def setModel(self, model):
        self.model = model
        self.n = len(model.ground_set)
        self.a = np.zeros(self.n)
        self.base = []

    def setBase(self, base):
        self.a = np.zeros(self.n)
        for i in base:
            self.a[i] = 1.0
        self.base = base

    def build(self):
        self.sample_count = 100
        self.n = len(self.model.ground_set)

        self.remaining = list(set(self.model.ground_set) - set(self.base))
        self.base_value = self.model.objective(self.base)
        # build B
        bases = self.model.matroid.bases
        self.m = len(bases)

        self.B = np.zeros(shape=(self.n, self.m))
        for i in range(0, self.m):
            for j in range(0, self.n):
                if j in bases[i]:
                    self.B[j, i] = 1

        self.nabla = self.gradient(self.a)
        for i in range(0, self.n):
            self.nabla[i] = -self.nabla[i]

        # print(f"b:{self.B.shape}, n:{self.nabla.shape}")
        self.final_weight = (np.transpose(self.B) @ self.nabla).reshape(1, -1)
        self.addition = self.F(self.a) - self.nabla @ self.a

    def optimize(self):
        # here we optimize x - a rather than x
        # thus the constraint should be A(x-a) <= b - Aa

        # bounds = np.array([(0, 1)] * self.n)
        bounds = [(0, 1)] * self.m

        unit_vector = np.ones(shape=(1, self.m))
        b = np.ones(shape=(1, 1))

        # print(f"f:{type(self.final_weight)}, {unit_vector.shape}, {b.shape}")

        x = scipy.optimize.linprog(c=self.final_weight, A_eq=unit_vector, b_eq=b, bounds=bounds).x

        v1 = - (self.final_weight @ x)
        v2 = self.addition

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])

        return {
            "upb": v1 + v2,
            "x": fs
        }
        pass


class MatroidOptimizer2:
    def __init__(self):
        self.eps = 0.1
        self.alpha = 0.8
        self.sample_count = 0

        self.model = None
        self.n = 0

        # start point
        self.a = None
        self.base = None
        self.w = None

        # the linear constraint
        self.L_c = None
        self.A = None
        self.b = None

        # the submodular constraint
        self.remaining = None
        self.NL_sub_c = None
        self.base_value = 0
        self.err = 0.01
        pass

    def F(self, x):
        total_value = 0
        for _ in range(0, self.sample_count):
            s = []
            for i in range(0, self.n):
                if random.random() < x[i]:
                    s.append(i)

            value = self.model.objective(s)
            total_value = total_value + value
        return total_value / self.sample_count

    def partial_derivative(self, x, i):
        total_value = 0
        for _ in range(0, self.sample_count):
            ground = list(set(self.model.ground_set) - {i})
            s = []
            for j in ground:
                if random.random() < x[j]:
                    s.append(j)

            value = self.model.objective(list(set(s) | {i})) - self.model.objective(s)
            total_value = total_value + value
        return total_value / self.sample_count

    def gradient(self, x):
        g = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(0, n):
            g[i] = self.partial_derivative(x=x, i=i)
        return g

    def evaluate_sample_count(self):
        nominator = 4 * math.log(1.0 / (1 - self.alpha), math.e)
        denominator = math.pow(self.eps, 2)
        return math.ceil(nominator / denominator)

    def setModel(self, model):
        self.model = model
        self.n = len(model.ground_set)
        self.a = np.zeros(self.n)
        self.base = []

    def setBase(self, base):
        self.a = np.zeros(self.n)
        for i in base:
            self.a[i] = 1.0
        self.base = base

    def sub_constraint(self, x):
        ret = np.zeros(self.n)

        for i in range(0, self.n):
            y = np.zeros(self.n)
            g_base_set = set()
            for j in range(0, i + 1):
                y[j] = x[j]
                if y[j] > 0:
                    g_base_set.add(j)

            item1 = self.w @ y
            item2 = self.model.objective(list(set(self.base) | g_base_set)) - self.base_value

            ret[i] = item1 + item2
        return ret

    def sub_constraint2(self, x):
        ret = np.zeros(self.n)

        for i in range(0, self.n):
            y = np.zeros(self.n)
            for j in range(0, i + 1):
                y[j] = x[j]

            item1 = self.w @ y
            item2 = 0
            prev = self.base_value
            g_base_set = set(self.base)
            for j in range(0, i + 1):
                if y[j] > 0:
                    g_base_set = g_base_set | {j}
                    item2 = item2 + (self.model.objective(list(g_base_set)) - prev) * y[j]
                    prev = self.model.objective(list(g_base_set))
            # print(f"item1:{item1}, item2:{item2}")
            ret[i] = item1 + item2
        return ret

    def sub_constraint_i(self, x, i):
        g_base_set = set()

        y = np.zeros(self.n)
        for j in range(0, i + 1):
            y[j] = x[j]
            if x[j] > 0:
                g_base_set.add(j)

        item1 = self.w @ y
        item2 = self.model.objective(list(set(self.base) | g_base_set)) - self.base_value

        ret = item1 + item2 + self.err

        return ret

    def sub_constraint_all(self, x):
        g_base_set = set()
        for i in range(0, self.n):
            if x[i] > 0:
                g_base_set.add(i)

        item1 = self.F(x)
        item2 = self.model.objective(list(set(self.base) | g_base_set)) - self.base_value

        ret = item2 - item1

        return ret

    def linear_constraint_i(self, x, i):
        item1 = (self.A[i] @ x)[0, 0]
        item2 = self.b[0, i]
        return item2 - item1

    def build(self):
        self.sample_count = 100
        self.remaining = list(set(self.model.ground_set) - set(self.base))
        self.base_value = self.model.objective(self.base)
        # build A
        self.A = self.model.A
        item1 = np.array(self.model.bv)
        item2 = self.A @ self.a
        self.b = np.asarray(item1 - item2).flatten()
        self.L_c = [
            scipy.optimize.LinearConstraint(A=self.A, lb=np.zeros(self.b.shape[0]), ub=self.b)
        ]
        self.NL_sub_c = [
            scipy.optimize.NonlinearConstraint(fun=self.sub_constraint, lb=np.zeros(self.n), ub=np.inf)
        ]

    def optimize(self):
        # here we optimize x - a rather than x
        # thus the constraint should be A(x-a) <= b - Aa

        # build w
        self.w = self.gradient(self.a)
        for i in range(0, self.n):
            self.w[i] = -self.w[i]

        # bounds = np.array([(0, 1)] * self.n)
        bounds = [(0, 1)] * self.n
        for i in self.base:
            bounds[i] = (0, 0)

        # build b
        for i in range(0, self.b.shape[0]):
            if self.b[i] < 0:
                return {
                    "upb": 10000,
                    "x": {}
                }

        x = scipy.optimize.minimize(lambda y: self.w @ y, x0=np.zeros(self.n), constraints=self.L_c + self.NL_sub_c,
                                    bounds=bounds).x

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])
        base = self.F(self.a)
        return {
            "upb": -(self.w @ x) + base,
            "x": fs
        }
        pass


class MultilinearDualOptimizer:
    def __init__(self):
        self.eps = 0.1
        self.alpha = 0.8
        self.sample_count = 0

        self.model = None
        self.n = 0

        # start point
        self.a = None
        self.base = None
        pass

    def F(self, x):
        total_value = 0
        for _ in range(0, self.sample_count):
            s = []
            for i in range(0, self.n):
                if random.random() < x[i]:
                    s.append(i)

            value = self.model.objective(s)
            total_value = total_value + value
        return total_value / self.sample_count

    def partial_derivative(self, x, i):
        total_value = 0
        for _ in range(0, self.sample_count):
            ground = list(set(self.model.ground_set) - {i})
            s = []
            for j in ground:
                if random.random() < x[j]:
                    s.append(j)

            value = self.model.objective(list(set(s) | {i})) - self.model.objective(s)
            total_value = total_value + value
        return total_value / self.sample_count

    def gradient(self, x):
        g = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(0, n):
            g[i] = self.partial_derivative(x=x, i=i)
        return g

    def evaluate_sample_count(self):
        nominator = 4 * math.log(1.0 / (1 - self.alpha), math.e)
        denominator = math.pow(self.eps, 2)
        return math.ceil(nominator / denominator)

    def setModel(self, model):
        self.model = model
        self.n = len(model.ground_set)
        self.a = np.zeros(self.n)
        self.base = []

    def setBase(self, base):
        self.a = np.zeros(self.n)
        for i in base:
            self.a[i] = 1.0
        self.base = base

    def build(self):
        self.sample_count = 100  # self.evaluate_sample_count()

    def optimize(self):
        # here we optimize x - a rather than x
        # thus the constraint should be A(x-a) <= b - Aa

        # build c
        c = np.zeros(self.n)
        for i in range(0, self.n):
            c[i] = self.model.cost_of_singleton(i)

        # build w
        # print(f"ground:{self.F(self.a)}, f:{self.model.objective(self.base)}, base:{self.base}")
        w = self.gradient(self.a)
        for i in range(0, self.n):
            w[i] = -w[i]
        w = np.array([w])

        bounds = np.array([(0, 1)] * self.n)
        for i in range(0, self.n):
            if i in self.base:
                bounds[i][1] = 0

        # build b
        b = np.array([self.model.objective(self.base) - self.model.value])

        x = scipy.optimize.linprog(c=c, A_ub=w, b_ub=b, bounds=bounds).x

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])

        return {
            "lwb": np.matmul(c, x),
            "x": fs
        }
        pass


class DualOptimizer:
    def __init__(self):
        self.model = None
        self.intermediate_sets = []
        self.upb = 'ub0'
        self.n = 0
        self.v = 0

        self.c = None
        self.A = None
        self.b = None

    def setModel(self, model):
        self.model = model

    def addIntermediate(self, intermediate):
        self.intermediate_sets.append(copy.deepcopy(intermediate))

    def setUpb(self, upb):
        self.upb = upb

    def build(self):
        # prepare c
        self.n = len(self.model.ground_set)

        self.c = np.zeros(self.n)
        for e in self.model.ground_set:
            self.c[e] = self.model.cost_of_singleton(e)

        # prepare A
        t = len(self.intermediate_sets)
        self.A = np.zeros(shape=(t, self.n))
        for r_idx in range(0, t):
            intermediate_set = list(self.intermediate_sets[r_idx])
            for e in range(0, self.n):
                self.A[r_idx, e] = -self.model.marginal_gain(e, intermediate_set)

        # prepare b
        self.b = np.zeros(t)
        self.v = self.model.value
        for r_idx in range(0, t):
            intermediate_set = list(self.intermediate_sets[r_idx])
            self.b[r_idx] = self.model.objective(intermediate_set) - self.v

    def optimize(self):
        bounds = [(0, 1)] * self.n

        # print(f"c:{self.c[:10]}, b:{self.b[:10]}, v:{self.model.value}, s:{self.model.objective(self.intermediate_sets[1])}")

        x = scipy.optimize.linprog(c=self.c, A_eq=self.A, b_eq=self.b, bounds=bounds).x

        # print(f"c:{self.c.shape}, x:{x.shape}")

        return {
            "lwb": self.c @ x
        }


class MaximizationOptimizer:
    def __init__(self):
        self.model = None
        self.intermediate_sets = []
        self.upb = 'ub0'
        self.n = 0

        self.c = None
        self.A = None
        self.b = None

    def setModel(self, model):
        self.model = model

    def addIntermediate(self, intermediate):
        self.intermediate_sets.append(copy.deepcopy(intermediate))

    def setUpb(self, upb):
        self.upb = upb

    def build(self):
        # prepare c
        self.n = len(self.model.ground_set)

        self.c = np.zeros(self.n + 1)
        self.c[self.n] = -1

        # prepare A

        # prepare lambda
        t = len(self.intermediate_sets)

        self.A = np.zeros(shape=(t + 1, self.n + 1))
        for r_idx in range(0, t):
            intermediate_set = list(self.intermediate_sets[r_idx])
            for e in range(0, self.n):
                self.A[r_idx, e] = -self.model.marginal_gain(e, intermediate_set)
            self.A[r_idx, self.n] = 1

        for e in range(0, self.n):
            self.A[t, e] = self.model.cost_of_singleton(e)

        # prepare b
        self.b = np.zeros(t + 1)
        for r_idx in range(0, t):
            intermediate_set = list(self.intermediate_sets[r_idx])
            self.b[r_idx] = self.model.objective(intermediate_set)
        self.b[t] = self.model.budget

    def optimize(self):
        bounds = [(0, 1)] * (self.n + 1)
        bounds[int(self.n)] = (0, np.inf)

        x = scipy.optimize.linprog(c=self.c, A_ub=self.A, b_ub=self.b, bounds=bounds).x
        # print(f"inter:{self.intermediate_sets}, A:{self.A}, b:{self.b}, x:{x}")
        return {
            "upb": -self.c @ x
        }


class PrimalDualOptimizer:
    def __init__(self):
        self.model = None
        self.b = 0
        self.n = 0
        self.tau = None
        self.x = None

        pass

    def build(self):
        self.b = self.model.budget
        self.n = len(self.model.ground_set)
        pass

    def optimize(self):
        i = 1
        # initial solution
        self.x = np.zeros(self.n)
        # fractional solution sequence
        x_series = [copy.deepcopy(self.x)]
        # map from indexes in x_series to weight
        self.tau = {
            0: 1,
        }
        c = 0
        T = set()

        d_gamma = 0
        betas = []
        for i in range(0, self.n):
            betas.append(self.beta(i))

        while c < self.b:
            self.discrete_make_dual()

            # find p_c
            p_c, t = None, 0
            for j, beta_j in T:
                if p_c is None or t < beta_j:
                    p_c = j
                    t = beta_j

            delta_c = max(self.b - c, self.model.cost_of_singleton(p_c))

            self.x[p_c] = self.x[p_c] + delta_c

            x_series.append(copy.deepcopy(self.x))



        pass

    def discrete_make_dual(self):
        pass

    def beta(self, j, base = None):
        if base is None:
            return self.density(j, self.x)
        else:
            return self.density(j, base)

    def density(self, j, base = None):
        pass

    def gain(self, j, base = None):
        pass
