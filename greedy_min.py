"""
# algorithm
# implement upper bound mentioned in revisiting original paper
"""
import time

from base_task import BaseTask
from data_dependent_upperbound import marginal_delta_min_gate


def simple_greedy_min(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    res = {}

    G = set()
    remaining_elements = set(model.ground_set)
    lambda_capital = 0.
    if upb is not None:
        delta, parameters = marginal_delta_min_gate(upb, set({}), remaining_elements, model)
        lambda_capital = delta

    while model.objective(list(G)) < model.value and len(remaining_elements) > 0:
        s, max_md = None, -1
        for e in remaining_elements:
            md = model.marginal_gain(e, list(G)) / model.cost_of_singleton(e)
            # print(f"e:{e}, md:{md}, s:{s}, max_md:{max_md}")
            if s is None or md > max_md:
                s, max_md = e, md
        temp_G = G | {s}
        if model.objective(list(temp_G)) < model.value:
            G.add(s)
        else:
            min_cost = model.cost_of_singleton(s)
            min_s = s
            for e in remaining_elements:
                temp_G = G | {e}
                if model.objective(list(temp_G)) >= model.value:
                    c_e = model.cost_of_singleton(e)
                    if c_e < min_cost:
                        min_s, min_cost = e, c_e
            G.add(min_s)

        if upb is not None:
            delta, parameters = marginal_delta_min_gate(upb, G, remaining_elements, model)
            if lambda_capital < delta:
                lambda_capital = delta

        remaining_elements.remove(s)


    res['S'] = G
    res['f(S)'] = model.objective(list(G))
    res['c(S)'] = model.cost_of_set(list(G))
    res['target'] = model.value
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['c(S)'] / lambda_capital

    stop_time = time.time()
    res['Time'] = stop_time - start_time
    return res


def augmented_greedy_min(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    res = {}

    G = set()
    remaining_elements = set(model.ground_set)
    lambda_capital = 0.
    if upb is not None:
        delta, parameters = marginal_delta_min_gate(upb, set({}), remaining_elements, model)
        lambda_capital = delta
        # print(f"l updated:{lambda_capital}, d:{delta} 1")

    while model.objective(list(G)) < model.value and len(remaining_elements) > 0:
        candidates = set()
        for e in remaining_elements:
            if model.objective(list(G | {e})) < model.value:
                candidates.add(e)

        if len(candidates) == 0:
            min_cost = -1
            min_s = None
            for e in remaining_elements:
                c_e = model.cost_of_singleton(e)
                if min_s is None or c_e < min_cost:
                    min_s, min_cost = e, c_e
            G.add(min_s)
        else:
            s, max_md = None, -1
            for e in candidates:
                md = model.marginal_gain(e, list(G)) / model.cost_of_singleton(e)
                if s is None or md > max_md:
                    s, max_md = e, md
            # print(f"1: s:{s}, md:{max_md}, md:{model.marginal_gain(419, list(G)) / model.cost_of_singleton(419)}")
            G.add(s)
            remaining_elements.remove(s)

        if upb is not None:
            delta, parameters = marginal_delta_min_gate(upb, G, remaining_elements, model)
            if lambda_capital < delta:
                lambda_capital = delta

    res['S'] = G
    res['f(S)'] = model.objective(list(G))
    res['c(S)'] = model.cost_of_set(list(G))
    res['target'] = model.value
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['c(S)'] / lambda_capital
        res['p'] = parameters

    stop_time = time.time()
    res['Time'] = stop_time - start_time
    return res
