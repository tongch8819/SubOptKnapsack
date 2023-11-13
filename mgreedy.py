from data_dependent_upperbound import marginal_delta
from data_dependent_upperbound import marginal_delta_version2
from data_dependent_upperbound import marginal_delta_version3


def modified_greedy(model):
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    while len(remaining_elements):
        #
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, sol)
            if u is None or ds > max_density:
                u, max_density = e, ds
        # TODO: filter out violating elements
        assert u is not None
        if cur_cost + model.costs_obj[u] <= model.budget:
            sol.add(u)
            cur_cost += model.costs_obj[u]
        remaining_elements.remove(u)

    v_star, v_star_fv = None, -1
    for e in model.ground_set:
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))
    if v_star_fv > sol_fv:
        return {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.costs_obj[v_star],
        }
    else:
        return {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
        }


def modified_greedy_ub1(model):
    """
    # algorithm
    # implement upper bound mentioned in revisiting original paper
    """
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    lambda_capital = float('inf')
    while len(remaining_elements):
        #
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, sol)
            if u is None or ds > max_density:
                u, max_density = e, ds
        # TODO: filter out violating elements
        assert u is not None
        if cur_cost + model.costs_obj[u] <= model.budget:
            sol.add(u)
            cur_cost += model.costs_obj[u]
            # update data-dependent upper-bound
            lambda_capital = min(lambda_capital, model.objective(
                sol) + marginal_delta(sol, remaining_elements - {u}, model))
        remaining_elements.remove(u)

    v_star, v_star_fv = None, -1
    for e in model.ground_set:
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))

    if v_star_fv > sol_fv:
        return {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.costs_obj[v_star],
            'Lambda': lambda_capital,
            'AF': v_star_fv / lambda_capital,
        }
    else:
        return {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
            'Lambda': lambda_capital,
            'AF': sol_fv / lambda_capital,
        }


def modified_greedy_ub2(model):
    """
    Modified Greedy Algorithm with Cardinality-like knapsack constraint
    """
    ground_set = set(model.ground_set)
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    lambda_capital = float('inf')
    while len(remaining_elements):
        #
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, sol)
            if u is None or ds > max_density:
                u, max_density = e, ds
        # TODO: filter out violating elements
        assert u is not None
        if cur_cost + model.costs_obj[u] <= model.budget:
            sol.add(u)
            cur_cost += model.costs_obj[u]
            # update data-dependent upper-bound
            lambda_capital = min(lambda_capital, model.objective(
                sol) + marginal_delta_version2(sol, remaining_elements - {u}, ground_set, model))
        remaining_elements.remove(u)

    v_star, v_star_fv = None, -1
    for e in model.ground_set:
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))
    # if v_star_fv > sol_fv:
    #     return set([v_star]), v_star_fv, lambda_capital
    # else:
    #     return sol, sol_fv, lambda_capital
    if v_star_fv > sol_fv:
        return {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.costs_obj[v_star],
            'Lambda': lambda_capital,
            'AF': v_star_fv / lambda_capital,
        }
    else:
        return {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
            'Lambda': lambda_capital,
            'AF': sol_fv / lambda_capital,
        }


def modified_greedy_ub3(model):
    """
    Modified Greedy Algorithm with Cardinality-like knapsack constraint
    """
    ground_set = set(model.ground_set)
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    lambda_capital = float('inf')
    while len(remaining_elements):
        #
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, sol)
            if u is None or ds > max_density:
                u, max_density = e, ds
        # TODO: filter out violating elements
        assert u is not None
        if cur_cost + model.costs_obj[u] <= model.budget:
            sol.add(u)
            cur_cost += model.costs_obj[u]
            # update data-dependent upper-bound
            lambda_capital = min(lambda_capital, model.objective(
                sol) + marginal_delta_version3(sol, remaining_elements - {u}, ground_set, model))
        remaining_elements.remove(u)

    v_star, v_star_fv = None, -1
    for e in model.ground_set:
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv

    sol_fv = model.objective(list(sol))
    # if v_star_fv > sol_fv:
    #     return set([v_star]), v_star_fv, lambda_capital
    # else:
    #     return sol, sol_fv, lambda_capital
    if v_star_fv > sol_fv:
        return {
            'S': [v_star],
            'f(S)': v_star_fv,
            'c(S)': model.costs_obj[v_star],
            'Lambda': lambda_capital,
            'AF': v_star_fv / lambda_capital,
        }
    else:
        return {
            'S': sol,
            'f(S)': sol_fv,
            'c(S)': cur_cost,
            'Lambda': lambda_capital,
            'AF': sol_fv / lambda_capital,
        }
