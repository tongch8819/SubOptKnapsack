import time

from base_task import BaseTask
from data_dependent_upperbound import marginal_delta
from data_dependent_upperbound import marginal_delta_version2
from data_dependent_upperbound import marginal_delta_version3
from data_dependent_upperbound import marginal_delta_version4
from data_dependent_upperbound import marginal_delta_gate
from copy import deepcopy

# OPT
# 下标typo
# it+是否可以大于it
# it=it+清晰


def greedy_max(model: BaseTask, upb: str = None):
    """
    # algorithm
    # implement upper bound mentioned in revisiting original paper
    """
    start_time = time.time()
    parameters = {}

    G, S = set(), set()
    remaining_elements = set(model.ground_set)
    # print(f"l:{len(remaining_elements)},s:{remaining_elements}")
    cur_cost = 0.
    if upb is not None:
        delta, parameters = marginal_delta_gate(upb, set({}), remaining_elements, model)
        lambda_capital = delta

    # def sort_key(x, y):
    #     if model.objective({x}) != model.objective({y}):
    #         return model.objective({x}) > model.objective({y})
    #     else:
    #         return model.cost_of_singleton(x) > model.cost_of_singleton(y)

    while len(remaining_elements):
        # argmax marginal gain
        s, max_marginal_gain = None, -1
        for e in remaining_elements:
            mg = model.marginal_gain(e, G)
            if s is None or mg > max_marginal_gain:
                s, max_marginal_gain = e, mg
        assert s is not None
        tmp_G = deepcopy(G)
        tmp_G.add(s)
        if model.objective(S) < model.objective(tmp_G) and model.cost_of_set(tmp_G) <= model.budget:
            S = tmp_G
            # update data-dependent upper-bound
            if upb is not None:
                delta, p1 = marginal_delta_gate(upb, S, set(model.ground_set) - S, model)
                fs = model.objective(S)
                if lambda_capital > fs + delta:
                    lambda_capital = fs + delta
                    parameters = p1

        # argmax density
        a, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, G)
            if a is None or ds > max_density:
                a, max_density = e, ds

        assert a is not None
        if cur_cost + model.cost_of_singleton(a) <= model.budget:
            G.add(a)
            cur_cost += model.cost_of_singleton(a)
            delta, p1 = marginal_delta_gate(upb, G, set(model.ground_set) - G, model)
            fs = model.objective(G)
            # if fs + delta < lambda_capital:
            #     print(f"new lambda:{fs + delta}, S:{S}, fs:{fs}, delta:{delta}")
            if lambda_capital > fs + delta:
                lambda_capital = fs + delta
                parameters = p1

        remaining_elements.remove(a)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        # for v in to_remove:
        #     remaining_elements.remove(v)
        remaining_elements -= to_remove

    S_fv = model.objective(S)
    G_fv = model.objective(G)
    if S_fv >= G_fv:
        res = {
            'S': S,
            'f(S)': S_fv,
            'c(S)': model.cost_of_set(S),
        }
    else:
        res = {
            'S': G,
            'f(S)': G_fv,
            'c(S)': model.cost_of_set(G),
        }

    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['f(S)'] / lambda_capital
        # res['ScanCount'] = parameters["ScanCount"]
        # res['MinusCount'] = parameters["MinusCount"]
        res['p'] = parameters

    stop_time = time.time()
    res['Time'] = stop_time - start_time

    return res


def greedy_max_ub1(model):
    return greedy_max(model, "ub1")

def greedy_max_ub1m(model):
    return greedy_max(model, "ub1m")

def greedy_max_ub2(model):
    return greedy_max(model, "ub2")


def greedy_max_ub3(model):
    return greedy_max(model, "ub3")


def greedy_max_ub4(model):
    return greedy_max(model, "ub4")


def greedy_max_ub4c(model):
    return greedy_max(model, "ub4c")


def greedy_max_ub4cm(model):
    return greedy_max(model, "ub4cm")


def greedy_max_ub5(model):
    return greedy_max(model, "ub5")


def greedy_max_ub5c(model):
    return greedy_max(model, "ub5c")


def greedy_max_ub5p(model):
    return greedy_max(model, "ub5p")


def greedy_max_ub6(model):
    return greedy_max(model, "ub6")


def greedy_max_ub7(model):
    return greedy_max(model, "ub7")


def greedy_max_ub7m(model):
    return greedy_max(model, "ub7m")


def greedy_max_ub8(model):
    return greedy_max(model, "ub8")


def greedy_max_ub9(model):
    return greedy_max(model, "ub9")


