import math
import random

import numpy as np

from base_task import BaseTask
from optimizer import PackingOptimizer, UpperBoundFunction


def MWU(model: BaseTask, upb=None, upb_function_mode='m1+'):
    S = set()
    n = len(model.ground_set)
    R = set(model.ground_set)

    A = model.A
    bv = model.bv
    m = A.shape[0]
    w = np.zeros(m)

    opt = PackingOptimizer()
    opt.setModel(model)
    opt.permutation_mode = 'none'

    if upb_function_mode == 'none':
        opt.upb_function = None
    else:
        upb_function = UpperBoundFunction(model.objective, model.ground_set)
        upb_function.setY(random.sample(model.ground_set, int(n/10)))
        upb_function.setType(upb_function_mode)
        upb_function.build()
        opt.upb_function = upb_function

    opt.build()
    # print(f"0 S:{opt.S.shape}")
    upper_bound_value = 0

    if upb == 'ub0':
        upper_bound_value = opt.optimize()['upb']
    elif upb == 'ub2':
        # opt.permutation_max()
        upper_bound_value = opt.optimize()['upb']

    W = -1
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            t = bv[i]/A[i, j]
            if W == -1 or t < W:
                W = t

    worst_case_guarantee = 1/(math.pow(m, 1/W))

    update_factor = math.exp(W) * m

    for i in range(0, m):
        w[i] = 1/bv[i]

    final_j = -1
    while True:
        m_budget = np.matmul(w, bv)
        if m_budget > update_factor:
            break
        if len(S) == n:
            break

        j = -1
        rule = -1
        no_zero = False
        for ele in R:
            marginal_ele = model.marginal_gain(ele, list(S))
            nominator = 0
            if marginal_ele != 0:
                for i in range(0, m):
                    # print(f"i:{i}, ele:{ele},c:{A[i, ele]}, w:{w[i]}, t:{A[i, ele]*w[i]}")
                    nominator += A[i, ele] * w[i]
                temp = nominator / marginal_ele
                if temp < rule or j == -1:
                    j = ele
                    rule = temp
                no_zero = True

        if not no_zero:
            break

        S.add(j)
        # print(f"forwarding...{S}")
        if upb == 'ub0':
            opt.setBase(S)
            temp = opt.optimize()['upb']
            if upper_bound_value > temp:
                upper_bound_value = temp
        elif upb == 'ub2':
            opt.setBase(S)
            opt.build()
            temp = opt.optimize()['upb']
            if upper_bound_value > temp:
                upper_bound_value = temp

        R.remove(j)
        final_j = j

        # print(f"A:{A.shape}, m:{m}, w:{w.shape}, j:{j}, c:{A[i, j]}")
        for i in range(0, m):
            w[i] = w[i] * math.pow(update_factor, A[i, j]/bv[i])

    x_s = np.zeros(n)
    for ele in S:
        x_s[ele] = 1

    constraint = np.matmul(A, x_s) <= bv
    obeying = True
    for i in range(0, constraint.shape[1]):
        c = constraint[0, i]
        obeying = obeying and c

    final_S = S

    if obeying:
        final_S = S
    else:
        candidate1 = S
        candidate1.remove(final_j)

        candidate2 = [final_j]

        if model.objective(list(candidate1)) >= model.objective(list(candidate2)):
            final_S = candidate1
        else:
            final_S = set(candidate2)

    return final_S, upper_bound_value, worst_case_guarantee

