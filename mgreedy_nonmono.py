from base_task import BaseTask
from nonmono_data_dependent_upperbound import singleton_knapsack_fill


def modified_greedy(model: BaseTask, upb : str = None):
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    if upb is not None:
        lambda_capital = float('inf')
    while len(remaining_elements):
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, sol)
            if u is None or ds > max_density:
                u, max_density = e, ds
        assert u is not None

        # positive greedy
        if max_density < 0.:
            break
        if cur_cost + model.cost_of_singleton(u) <= model.budget:
            # satisfy the knapsack constraint
            sol.add(u)
            cur_cost += model.cost_of_singleton(u)

            # update data-dependent upper-bound
            if upb is not None:
                if upb == "ub1":
                    lambda_capital = singleton_knapsack_fill(model)
                else:
                    raise ValueError("Unsupported Upperbound")
                
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

    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['f(S)'] / lambda_capital
    return res

def modified_greedy_ub1(model: BaseTask):
    return modified_greedy(model, "ub1")

