from base_task import BaseTask
from algorithms.data_dependent_upperbound import marginal_delta
from algorithms.data_dependent_upperbound import marginal_delta_version2
from algorithms.data_dependent_upperbound import marginal_delta_version3
from copy import deepcopy


def greedy_max(model: BaseTask, upb: str = None):
    """
    # algorithm
    # implement upper bound mentioned in revisiting original paper
    """
    G, S = set(), set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    if upb is not None:
        lambda_capital = float('inf')
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
                if upb == "ub1":
                    delta = marginal_delta(S, remaining_elements - {s}, model)
                    fs = model.objective(S)
                    lambda_capital = min(lambda_capital, fs + delta)
                elif upb == "ub2":
                    delta = marginal_delta_version2(
                        S, remaining_elements - {s}, model)
                    fs = model.objective(S)
                    lambda_capital = min(lambda_capital, fs + delta)
                elif upb == "ub3":
                    delta = marginal_delta_version3(
                        S, remaining_elements - {s}, model)
                    fs = model.objective(S)
                    lambda_capital = min(lambda_capital, fs + delta)
                else:
                    raise ValueError("Unsupported Upperbound")

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

        remaining_elements.remove(a)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
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
    return res


def greedy_max_ub1(model):
    return greedy_max(model, "ub1")


def greedy_max_ub2(model):
    return greedy_max(model, "ub2")


def greedy_max_ub3(model):
    return greedy_max(model, "ub3")
