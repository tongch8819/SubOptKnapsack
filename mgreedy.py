from base_task import BaseTask

from data_dependent_upperbound import marginal_delta, marginal_delta_version4
from data_dependent_upperbound import marginal_delta_version2
from data_dependent_upperbound import marginal_delta_version3
from data_dependent_upperbound import marginal_delta_gate


def modified_greedy(model: BaseTask, upb : str = None):
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    if upb is not None:
        delta = marginal_delta_gate(upb, set({}), remaining_elements, model)
        # if fs + delta < lambda_capital:
        #     print(f"new lambda:{fs + delta}, S:{S}, fs:{fs}, delta:{delta}")
        lambda_capital = delta

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
            delta = marginal_delta_gate(upb, sol, remaining_elements - {u}, model)
            fs = model.objective(sol)
            lambda_capital = min(lambda_capital, fs + delta)
            '''
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
            '''
                
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