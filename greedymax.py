from data_dependent_upperbound import marginal_delta
from data_dependent_upperbound import marginal_delta_version2
from data_dependent_upperbound import marginal_delta_version3
from copy import deepcopy


def greedy_max_ub1(model):
    """
    # algorithm
    # implement upper bound mentioned in revisiting original paper
    """
    G, S = set(), set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
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
            lambda_capital = min(lambda_capital, model.objective(S) + marginal_delta(S, remaining_elements - {s}, model))

        # argmax density
        a, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, G)
            if a is None or ds > max_density:
                a, max_density = e, ds
        # TODO: filter out violating elements
        assert a is not None
        if cur_cost + model.costs_obj[a] <= model.budget:
            G.add(a)
            cur_cost += model.costs_obj[a]

        remaining_elements.remove(a)

    S_fv = model.objective(S)
    G_fv = model.objective(G)
    if S_fv >= G_fv:
        return {
            'S': S,
            'f(S)': S_fv,
            'c(S)': model.cost_of_set(S),
            'Lambda': lambda_capital,
            'AF': S_fv / lambda_capital,
        }
    else:
        return {
            'S': G,
            'f(S)': G_fv,
            'c(S)': model.cost_of_set(G),
            'Lambda': lambda_capital,
            'AF': G_fv / lambda_capital,
        }


def greedy_max_ub2(model):
    """
    # algorithm
    # implement upper bound mentioned in revisiting original paper
    """
    G, S = set(), set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    lambda_capital = float('inf')
    while len(remaining_elements):
        # argmax marginal gain
        s, max_marginal_gain = None, -1
        for e in remaining_elements:
            mg = model.marginal_gain(e, G)
            if mg > max_marginal_gain:
                s, max_marginal_gain = e, mg
        assert s is not None
        tmp_G = deepcopy(G)
        tmp_G.add(s)
        if model.objective(S) < model.objective(tmp_G) and model.cost_of_set(tmp_G) <= model.budget:
            S = tmp_G
            # update data-dependent upper-bound
            lambda_capital = min(lambda_capital, model.objective(S) + marginal_delta_version2(S, remaining_elements - {s}, set(model.ground_set), model))

        # argmax density
        a, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, G)
            if ds > max_density:
                a, max_density = e, ds
        # TODO: filter out violating elements
        assert a is not None
        if cur_cost + model.costs_obj[a] <= model.budget:
            G.add(a)
            cur_cost += model.costs_obj[a]

        remaining_elements.remove(a)

    S_fv = model.objective(S)
    G_fv = model.objective(G)
    if S_fv >= G_fv:
        return {
            'S': S,
            'f(S)': S_fv,
            'c(S)': model.cost_of_set(S),
            'Lambda': lambda_capital,
            'AF': S_fv / lambda_capital,
        }
    else:
        return {
            'S': G,
            'f(S)': G_fv,
            'c(S)': model.cost_of_set(G),
            'Lambda': lambda_capital,
            'AF': G_fv / lambda_capital,
        }
        


def greedy_max_ub3(model):
    """
    # algorithm
    # implement upper bound mentioned in revisiting original paper
    """
    G, S = set(), set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    lambda_capital = float('inf')
    while len(remaining_elements):
        # argmax marginal gain
        s, max_marginal_gain = None, -1
        for e in remaining_elements:
            mg = model.marginal_gain(e, G)
            if mg > max_marginal_gain:
                s, max_marginal_gain = e, mg
        assert s is not None
        tmp_G = deepcopy(G)
        tmp_G.add(s)
        if model.objective(S) < model.objective(tmp_G) and model.cost_of_set(tmp_G) <= model.budget:
            S = tmp_G
            # update data-dependent upper-bound
            remaining_elements.remove(s)
            
            lambda_capital = min(lambda_capital, model.objective(S) + marginal_delta_version3(S, remaining_elements, set(model.ground_set), model))

        # argmax density
        a, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, G)
            if a is None or ds > max_density:
                a, max_density = e, ds
        # TODO: filter out violating elements
        assert a is not None
        if cur_cost + model.costs_obj[a] <= model.budget:
            G.add(a)
            cur_cost += model.costs_obj[a]

        remaining_elements.remove(a)

    S_fv = model.objective(S)
    G_fv = model.objective(G)
    if S_fv >= G_fv:
        return {
            'S': S,
            'f(S)': S_fv,
            'c(S)': model.cost_of_set(S),
            'Lambda': lambda_capital,
            'AF': S_fv / lambda_capital,
        }
    else:
        return {
            'S': G,
            'f(S)': G_fv,
            'c(S)': model.cost_of_set(G),
            'Lambda': lambda_capital,
            'AF': G_fv / lambda_capital,
        }
