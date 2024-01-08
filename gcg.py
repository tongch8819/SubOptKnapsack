import copy
import math
import random
from itertools import accumulate

import numpy as np

from base_task import BaseTask
from data_dependent_upperbound import marginal_delta, marginal_delta_version2, marginal_delta_version3, \
    marginal_delta_version4


def get_upb(model: BaseTask, upb: str):
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    lambda_capital = float('inf')

    if upb == "ub0":
        return max([model.objective(x) for x in remaining_elements]) * len(remaining_elements)

    while len(remaining_elements):
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, sol)
            if u is None or ds > max_density:
                u, max_density = e, ds
        assert u is not None
        if cur_cost + model.cost_of_singleton(u) <= model.budget:
            # satisfy the knapsack constraint
            sol.add(u)
            cur_cost += model.cost_of_singleton(u)

            # update data-dependent upper-bound
            if upb is not None:
                if upb == "ub1":
                    delta = marginal_delta(sol, remaining_elements - {u}, model)
                    fs = model.objective(sol)
                    lambda_capital = min(lambda_capital, fs + delta)
                elif upb == "ub2":
                    delta = marginal_delta_version2(sol, remaining_elements - {u}, model)
                    fs = model.objective(sol)
                    lambda_capital = min(lambda_capital, fs + delta)
                elif upb == "ub3":
                    delta = marginal_delta_version3(sol, remaining_elements - {u}, model)
                    fs = model.objective(sol)
                    lambda_capital = min(lambda_capital, fs + delta)
                elif upb == 'ub4':
                    delta = marginal_delta_version4(sol, remaining_elements - {u}, model)
                    fs = model.objective(sol)
                    lambda_capital = min(lambda_capital, fs + delta)
                else:
                    raise ValueError("Unsupported Upperbound")

        remaining_elements.remove(u)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    return lambda_capital

def decreasing_density(model: BaseTask, eps, guessed_value, x):
    n = len(model.ground_set)

    w_e = np.zeros(n)

    sample_count = n * math.log(n) / pow(eps, 3)
    sample_count_2 = n * math.log(n) / pow(eps, 2)


    v_count = np.zeros(n)
    reinsert_limitation = math.log(n) / eps

    Q = []

    for e in model.ground_set:
        w_e[e] = estimate_margin(x, e, sample_count)

        v_e_threshold = w_e[e] / model.cost_of_singleton(e)
        temp_v = 1
        while temp_v > v_e_threshold:
            temp_v = temp_v * (1 - eps)

        Q.append((e, temp_v))

    Q.sort(key=lambda q: q[1], reverse=True)

    S = []
    x_s = x

    while len(Q) > 0:
        top_q = Q.pop(0)
        w_e_s = estimate_margin(model, x_s, top_q[0], sample_count_2)

        v_e_threshold = w_e_s / model.cost_of_singleton(top_q[0])
        temp_v = 1
        while temp_v > v_e_threshold:
            temp_v = temp_v * (1 - eps)

        if temp_v >= (1-eps) * top_q[1]:
            S.append(top_q[0])
            x_s[top_q[0]] = 1
        elif v_count[top_q[0]] < reinsert_limitation:
            idx = 0
            while idx < len(Q) and Q[idx][1] > temp_v:
                idx = idx + 1
            top_q[1] = temp_v
            Q.insert(idx + 1, top_q)

        total_margin = estimate_margin_eps(model, x, eps, S, sample_count)
        if total_margin >= (1- 10 * eps) * guessed_value:
            break

    return S

def rounding(model: BaseTask, eps, y , z):
    x = copy.deepcopy(z)

    z_prime = copy.deepcopy(z)

    for y_i in y:
        x = x + y_i

    S_l = set()
    S_s = set()

    for i in range(0, y.shape[0]):
        total_v = np.sum(y[i])
        standard = random.random() * total_v
        e_i = 0

        prefix_i = np.array(list(accumulate(y[i])))

        while prefix_i[e_i] < standard:
            e_i += 1
        S_l.add(e_i)

    z_max = np.sum([z[i] * model.objective(i) for i in range(0, z.shape[0])])

    for i in range(0, z.shape[0]):
        if model.cost_of_singleton(i) < pow(eps, 3) * z_max:
            z_prime[i] = z[i]
        else:
            z_prime[i] = 0

    return S_l | S_s


def sample(model: BaseTask, x):
    s = []
    for i in range(0, len(x)):
        if random.random() < x[i]:
            s.append(i)
    return s


def estimate_func(model: BaseTask, x, sample_count):
    return np.mean([model.objective(sample(model, x)) for i in range(0, sample_count)])


def estimate_margin_eps(model: BaseTask, x, eps, S, sample_count):
    x_prime = copy.deepcopy(x)
    for s in S:
        x_prime[s] = x_prime[s] + eps
    return estimate_func(model, x_prime, sample_count) - estimate_func(model, x, sample_count)


def estimate_margin(model: BaseTask, x, e, sample_count):
    estimation = []
    for i in range(0, sample_count):
        estimation.append(model.marginal_gain(e, sample(model, x)))
    return np.mean(estimation)


def subroutine(model: BaseTask, eps, w, W, duplicate_count):
    n = len(model.ground_set)

    sample_count = int(n * math.log(n) / pow(eps, 2))

    print(f"sample:{sample_count}")

    x = np.zeros(n)

    y = np.zeros((duplicate_count, n))

    z = np.zeros(n)

    t = eps

    while t < 1:
        for j in range(0, duplicate_count):
            candidate = None
            candidate_cost = float('inf')
            for e in model.ground_set:
                if estimate_margin(model, x, e, sample_count) >= w[j] and model.cost_of_singleton(e) < candidate_cost:
                    candidate = e
                    candidate_cost = model.cost_of_singleton(e)
            if candidate is not None:
                x[candidate] = x[candidate] + eps
                y[j][candidate] = y[j][candidate] + eps

        V = decreasing_density(model, eps, W, x)
        z = z + eps * V
        x = x + eps * V

        t = t + eps

    S = rounding(model, x, y, z)

    if model.cost_of_set(list(S)) > model.budget:
        S = []
    return list(S)


def dfs(model, eps, w, j, guess_set, duplicate_count, res):
    if j < duplicate_count:
        for i in range(0,len(guess_set)):
            w[j] = guess_set[i]
            dfs(model, eps, w, j + 1, guess_set, duplicate_count, res)
    else:
        for i in range(0,len(guess_set)):
            W = guess_set[i]
            S = subroutine(model, eps, w, W, duplicate_count)
            if model.objective(S) > res['f(S)']:
                res['f(S)'] = model.objective(S)
                res['S'] = list(S)


def gcg(model: BaseTask, upb: str = "ub0", eps = 0.5):
    # determine the threshold
    d = get_upb(model, upb)

    duplicate_count = int(1/pow(eps, 6))

    guess_set = []

    temp_d = d

    # prepare guess set
    guess_threshold = eps * eps * d

    print(duplicate_count)

    while temp_d > guess_threshold / len(model.ground_set):
        guess_set.append(temp_d)
        temp_d = temp_d * (1-eps)

    guess_set.append(guess_threshold)
    guess_set.append(0)

    time_set = []
    temp = eps
    while temp < 1:
        time_set.append(temp)
        temp += eps

    w = np.zeros(duplicate_count)

    res = {
        'f(S)' : float('-inf'),
        'S' : []
    }

    for t in time_set:
        dfs(model, eps, w, 0, guess_set, duplicate_count, res)

    return res

def gcg_ub1(model: BaseTask):
    return gcg(model, "ub1")


def gcg_ub2(model: BaseTask):
    return gcg(model, "ub2")


def gcg_ub3(model: BaseTask):
    return gcg(model, "ub3")


def gcg_ub4(model: BaseTask):
    return gcg(model, "ub4")
