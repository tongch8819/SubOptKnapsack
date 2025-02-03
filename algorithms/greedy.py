from application.base_task import BaseTask

def greedy_density_knapsack(model: BaseTask):
    budget = model.constraint.budget
    cost_func = model.constraint.cost_func

    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    while len(remaining_elements):
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, list(sol))
            if u is None or ds > max_density:
                u, max_density = e, ds
        assert u is not None
        if cur_cost + cost_func[u] <= budget:
            sol.add(u)
            cur_cost += cost_func[u]
        remaining_elements.remove(u)
        to_remove = set()
        for v in remaining_elements:
            if cost_func[v] + cur_cost > budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    res = {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': cur_cost,
    }
    return res


def greedy_marginal_gain_knapsack(model: BaseTask):
    budget = model.constraint.budget
    cost_func = model.constraint.cost_func

    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    while len(remaining_elements):
        u, max_marginal_gain = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.marginal_gain(e, list(sol))
            if u is None or ds > max_marginal_gain:
                u, max_marginal_gain = e, ds
        assert u is not None
        if cur_cost + cost_func[u] <= budget:
            sol.add(u)
            cur_cost += cost_func[u]
        remaining_elements.remove(u)
        to_remove = set()
        for v in remaining_elements:
            if cost_func[v] + cur_cost > budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    res = {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': cur_cost,
    }
    return res