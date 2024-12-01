import copy
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
        # update d and fA
        m = len(self.bv)
        n = len(self.remaining)

        self.base_value = self.model.objective(list(self.base))
        self.remaining = list(set(self.model.ground_set) - set(self.base))
        self.remaining.sort(key=self.f_s, reverse=True)

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

        # print(f"x:{x}")
        # print(f"d:{self.d}")
        # print(f"A:{self.fA}")
        # # print(f"bv:{self.model.bv}")
        # # print(f"total budget:{np.sum(x[n: 2*n])}")
        # print(f"r:{self.remaining}")
        # print(f"base:{self.base}, base v:{self.base_value}")
        # print(f"l1:{-np.matmul(w, x)}")
        # print(f"l2:{-np.matmul(w, x)+ self.base_value}")


        # second phase
        vti_dict = {}
        remaining_plus = self.remaining
        for e_idx in range(0, n):
            vti_dict[remaining_plus[e_idx]] = e_idx

        remaining_plus = list(set(self.model.ground_set) - set(self.base))
        remaining_plus.sort(key=lambda ele: x[vti_dict[ele]], reverse=True)

        # print(f"re:{remaining_plus}, v:{[x[vti_dict[ele]] for ele in remaining_plus]}")

        fA_plus = np.zeros(n)
        for e_idx in range(0, n):
            fA_plus[e_idx] = self.f_s(remaining_plus[:e_idx + 1])

        self.A_plus = self.model.A[:, list(remaining_plus)]

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
                A[c_idx, e_idx + n] = self.A_plus[c_idx, e_idx]
                A[c_idx, additive_idx] = -self.model.bv[c_idx]

        # condition (iii)
        # this type of constraints occupies the following n rows

        for e_idx in range(0, n):
            # row of this constraint in A
            r = e_idx + m
            # v_i
            A[r, e_idx] = 1
            # s_i
            A[r, e_idx + n] = -self.f_s({remaining_plus[e_idx]})

        # condition (iv)
        # this type of constraints occupies the following n rows

        start_row = m + n
        for c_iii_offset in range(0, n):
            r = start_row + c_iii_offset
            for v_idx in range(0, c_iii_offset + 1):
                A[r, v_idx] = 1
            A[r][additive_idx] = -fA_plus[c_iii_offset]
        # print(f"A:{A}")

        x = scipy.optimize.linprog(c=w, A_ub=A, b_ub=b, bounds=bounds).x

        # print(f"x plus:{x}")
        # print(f"fA plus:{fA_plus}")
        # print(f"single plus:{[self.f_s(ele) for ele in remaining_plus]}")

        fs = {}
        for i in range(0, len(x)):
            if x[i] > 0:
                fs[i] = float(x[i])

        return {
            "upb": -np.matmul(w, x) + self.base_value,
            "x": fs
        }
