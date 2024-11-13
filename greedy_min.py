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
        # print(f"l updated:{lambda_capital}, d:{delta} 1")

    while model.objective(list(G)) < model.value:
        s, max_marginal_gain = None, -1
        for e in remaining_elements:
            mg = model.marginal_gain(e, G)
            if s is None or mg > max_marginal_gain:
                s, max_marginal_gain = e, mg

        G.add(s)
        if upb is not None:
            delta, parameters = marginal_delta_min_gate(upb, G, remaining_elements, model)
            if lambda_capital < delta:
                lambda_capital = delta

        remaining_elements.remove(s)

    res['S'] = G
    res['c(S)'] = model.cost_of_set(list(G))
    res['target'] = model.value
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['c(S)'] / lambda_capital
        res['p'] = parameters

    stop_time = time.time()
    res['Time'] = stop_time - start_time

    return res