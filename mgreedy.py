import copy
import time

import optimizer
from base_task import BaseTask

from data_dependent_upperbound import marginal_delta, marginal_delta_version4
from data_dependent_upperbound import marginal_delta_version2
from data_dependent_upperbound import marginal_delta_version3
from data_dependent_upperbound import marginal_delta_gate


def modified_greedy(model: BaseTask, upb: str = None):
    start_time = time.time()

    sol = set()
    remaining_elements = set(model.ground_set)
    ground_set = set(model.ground_set)
    cur_cost = 0.
    parameters = {}

    updated = False

    update_upb = True

    if upb is not None:
        delta, p1 = marginal_delta_gate(upb, set({}), ground_set, model)
        lambda_capital = delta
        parameters = p1

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

            delta, p1 = marginal_delta_gate(upb, sol, ground_set - sol, model)
            fs = model.objective(sol)

            if lambda_capital > fs + delta and update_upb:
                lambda_capital = fs + delta
                parameters = p1
                updated = True

        remaining_elements.remove(u)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    # find the maximum singleton
    v_star, v_star_fv = None, float('-inf')
    for e in model.ground_set:
        if model.cost_of_singleton(e) > model.budget:
            # filter out singleton whose cost is larger than budget
            continue
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))

    if v_star_fv > sol_fv:
        res = {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.cost_of_singleton(v_star),
        }
    else:
        res = {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
        }

    print(model.budget)
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['f(S)'] / lambda_capital
        res['parameters'] = parameters
        res['updated'] = updated

    stop_time = time.time()
    res['Time'] = stop_time - start_time

    return res


def modified_greedy_nis(model: BaseTask, upb: str = None):
    start_time = time.time()

    sol = set()
    remaining_elements = set(model.ground_set)
    ground_set = set(model.ground_set)
    cur_cost = 0.
    parameters = {}

    # print("I am here!")

    lambda_capital = 0

    # if upb is not None:
    #     delta, p1 = marginal_delta_gate(upb, set({}), ground_set, model)
    #     lambda_capital = delta
    #     parameters = p1

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

        # print(f"sol:{sol}, u:{u}, budget:{cur_cost}")

        remaining_elements.remove(u)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    # find the maximum singleton
    v_star, v_star_fv = None, float('-inf')
    for e in model.ground_set:
        if model.cost_of_singleton(e) > model.budget:
            # filter out singleton whose cost is larger than budget
            continue
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))

    if upb is not None:
        delta, p1 = marginal_delta_gate(upb, sol, ground_set - sol, model)
        lambda_capital = sol_fv + delta
        parameters = p1

    if v_star_fv > sol_fv:
        res = {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.cost_of_singleton(v_star),
        }
        print(f"Single is selected. Single:{v_star}, G:{sol}")
    else:
        res = {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
        }
        print(f"G is selected. Single:{v_star}, G:{sol}")
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['f(S)'] / lambda_capital
        res['parameters'] = parameters

    stop_time = time.time()
    res['Time'] = stop_time - start_time

    return res

def modified_greedy_ub10(model: BaseTask):
    start_time = time.time()

    sol = set()
    remaining_elements = set(model.ground_set)
    ground_set = set(model.ground_set)
    cur_cost = 0.
    parameters = {}
    upb = 'ub1'

    updated = False

    delta, p1 = marginal_delta_gate(upb, set({}), ground_set, model)
    prev_ub1 = delta
    beta = 1
    betas = [1]
    bases = [[]]

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
            mg = model.marginal_gain(u, list(sol))
            sol.add(u)

            cur_cost += model.cost_of_singleton(u)
            delta, p1 = marginal_delta_gate(upb, sol, ground_set - sol, model)

            beta = beta * (1 - (mg / prev_ub1))
            betas.append(beta)
            bases.append(copy.deepcopy(sol))
            # print(f"beta:{beta}, mg:{mg}, prev:{prev_ub1}")

            fs = model.objective(list(sol))
            prev_ub1 = fs + delta

        remaining_elements.remove(u)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    # find the maximum singleton
    v_star, v_star_fv = None, float('-inf')
    for e in model.ground_set:
        if model.cost_of_singleton(e) > model.budget:
            # filter out singleton whose cost is larger than budget
            continue
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))

    if v_star_fv > sol_fv:
        res = {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.cost_of_singleton(v_star),
        }
    else:
        res = {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
        }

    lambda_capital = sol_fv / (1 - beta)

    for i in range(0, len(betas) - 1):
        temp_beta = beta/betas[i]
        mg = sol_fv - model.objective(bases[i])
        temp_lambda = mg / (1 - temp_beta)
        # print(f"i:{i}, temp_b:{temp_beta}, mg:{mg}, bases:{bases[i]}, temp_l:{temp_lambda}, S:{sol}")
        if temp_lambda < lambda_capital:
            lambda_capital = temp_lambda

    res['Lambda'] = lambda_capital
    res['AF'] = res['f(S)'] / lambda_capital
    res['parameters'] = parameters
    res['updated'] = updated

    stop_time = time.time()
    res['Time'] = stop_time - start_time

    return res


def modified_greedy_ub11(model: BaseTask):
    start_time = time.time()

    sol = set()
    remaining_elements = set(model.ground_set)
    ground_set = set(model.ground_set)
    cur_cost = 0.
    parameters = {}

    updated = False

    update_upb = True

    opt = optimizer.MaximizationOptimizer()
    opt.setModel(model)
    opt.addIntermediate(set())

    while len(remaining_elements):
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, list(sol))
            if u is None or ds > max_density:
                u, max_density = e, ds
        assert u is not None
        if cur_cost + model.cost_of_singleton(u) <= model.budget:
            # satisfy the knapsack constraint
            sol.add(u)
            opt.addIntermediate(sol)
            cur_cost += model.cost_of_singleton(u)

        remaining_elements.remove(u)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    # find the maximum singleton
    v_star, v_star_fv = None, float('-inf')
    for e in model.ground_set:
        if model.cost_of_singleton(e) > model.budget:
            # filter out singleton whose cost is larger than budget
            continue
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))

    if v_star_fv > sol_fv:
        res = {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.cost_of_singleton(v_star),
        }
    else:
        res = {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
        }

    opt.build()
    lambda_capital = opt.optimize()['upb']

    res['Lambda'] = lambda_capital
    res['AF'] = res['f(S)'] / lambda_capital
    res['parameters'] = parameters
    res['updated'] = updated

    stop_time = time.time()
    res['Time'] = stop_time - start_time

    return res

def modified_greedy_plain(model: BaseTask):
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.

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

        remaining_elements.remove(u)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    # find the maximum singleton
    v_star, v_star_fv = None, float('-inf')
    for e in model.ground_set:
        if model.cost_of_singleton(e) > model.budget:
            # filter out singleton whose cost is larger than budget
            continue
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))

    if v_star_fv > sol_fv:
        res = {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.cost_of_singleton(v_star),
        }
    else:
        res = {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
        }

    return res


def modified_greedy_ub1(model: BaseTask):
    return modified_greedy(model, "ub1")


def modified_greedy_ub2(model: BaseTask):
    return modified_greedy(model, "ub2")


def modified_greedy_ub3(model: BaseTask):
    return modified_greedy(model, "ub3")


def modified_greedy_ub4(model: BaseTask):
    return modified_greedy(model, "ub4")


def modified_greedy_ub4c(model):
    return modified_greedy(model, "ub4c")


def modified_greedy_ub4cm(model):
    return modified_greedy(model, "ub4cm")


def modified_greedy_ub5(model: BaseTask):
    return modified_greedy(model, "ub5")


def modified_greedy_ub5c(model: BaseTask):
    return modified_greedy(model, "ub5c")


def modified_greedy_ub5p(model: BaseTask):
    return modified_greedy(model, "ub5p")


def modified_greedy_ub6(model: BaseTask):
    return modified_greedy(model, "ub6")


def modified_greedy_ub7(model: BaseTask):
    return modified_greedy(model, "ub7")


def modified_greedy_ub7m(model: BaseTask):
    return modified_greedy(model, "ub7m")


def modified_greedy_ub1m(model: BaseTask):
    return modified_greedy(model, "ub1m")


def modified_greedy_ub8(model: BaseTask):
    return modified_greedy(model, "ub8")


def modified_greedy_ub9(model: BaseTask):
    return modified_greedy(model, "ub9")


def modified_greedy_nis_ub1(model: BaseTask):
    return modified_greedy_nis(model, "ub1")


def modified_greedy_nis_ub1m(model: BaseTask):
    return modified_greedy_nis(model, "ub1m")


def modified_greedy_nis_ub7(model: BaseTask):
    return modified_greedy_nis(model, "ub7")


def modified_greedy_nis_ub7m(model: BaseTask):
    return modified_greedy_nis(model, "ub7m")


def greedy_heuristic_for_matroid(model: BaseTask, upb: str):
    model.enable_matroid()
    s = set()
    v = set(model.ground_set)

    while len(v) > 0:
        max_ele = None
        max_marginal_value = -1

        for ele in v:
            if model.marginal_gain(ele, list(s)) > max_marginal_value:
                max_marginal_value = model.marginal_gain(ele, list(s))
                max_ele = ele

        if model.matroid.is_legal(s | {max_ele}):
            s = s | {max_ele}

        v.remove(max_ele)

    max_cardinality = len(s)
    model.budget = max_cardinality
    lc = modified_greedy(model, upb)["Lambda"]

    res = {
        "S": s,
        "f(S)": model.objective(s),
        "upb": lc,
        "AF": model.objective(s) / lc
    }

    return res
