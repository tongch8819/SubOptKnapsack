import random

import estimator
from base_task import BaseTask
import numpy as np


def decreasing_threshold(model: BaseTask, x, eps, r):
    B = set()
    d = np.max([model.objective(ele) for ele in model.ground_set])
    w = d
    s = r * np.log2(len(model.ground_set)) / (eps * eps)
    while w >= eps * d / r:
        for ele in model.ground_set:
            if model.matroid.is_legal(B | {ele}):
                xb = x
                for key in B:
                    xb[key] = min(xb[key] + eps, 1)
                w_e = estimator.estimate_marginal_gain(model, xb, ele, s)
                if w_e >= w:
                    B = B | {ele}
        w = (1 - eps) * w
    return B


def swap_rounding(model: BaseTask, x):
    pass


def hit_constraint(model: BaseTask, y, i, j):
    def rmy(A):
        rank = model.matroid.r(A)
        ry = 0
        for k in y.keys():
            ry += y[k]
        return rank - ry
    pass
def is_integral(x):
    for k in x.keys():
        xi = x[k]
        if xi not in {0, 1}:
            return False
    return True


def pick_fractional(T, x):
    ret = []

    for k in x.keys() and len(ret) < 2:
        xi = x[k]
        if (xi in T) and (xi not in {0, 1}):
            ret.append(xi)

    return ret

def distance(x, y):
    total = 0
    all_keys = set(x.keys()) | set(y.keys())
    for key in all_keys:
        if key in x.keys() and key in y.keys():
            total += pow(x[key] - y[key], 2)
        elif key in x.keys():
            total += pow(x[key], 2)
        elif key in y.keys():
            total += pow(y[key], 2)
    return float(np.sqrt(total))

def pipage_rounding(model: BaseTask, y):
    while not is_integral(y):
        T = set(model.ground_set)
        fv = pick_fractional(T, y)
        while len(fv) > 0:
            i = fv[0]
            j = fv[1]
            y_p, A_p = hit_constraint(model, y, i, j)
            y_m, A_m = hit_constraint(model, y, i, j)
            if y == y_p == y_m:
                T = T.intersection(A_p)
            else:
                nominator = distance(y_p, y)
                denominator = distance(y_p, y_m)
                p = nominator/denominator
                t = random.random()
                if t < p:
                    y = y_m
                    T = T.intersection(A_m)
                else:
                    y = y_p
                    T = T.intersection(A_p)

            fv = pick_fractional(T, y)

    ret = set()
    for key in y.keys():
        if y[key] > 0:
            ret.add(key)

    return ret


def acgm(model: BaseTask, eps, r):
    x = {}
    t = 0
    while t <= 1:
        t += eps
        B = decreasing_threshold(model, x, eps, r)
        for key in B:
            x[key] = min(x[key] + eps, 1)

    S = pipage_rounding(model, x)
    return S
