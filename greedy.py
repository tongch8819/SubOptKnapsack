from base_task import BaseTask


def greedy(model: BaseTask):
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    while len(remaining_elements):
        #
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            # mg = marginal_gain(objective, e, sol)
            # density = (mg * 100100) / (costs_obj[e] * 100100)  # cost may be too small
            ds = model.density(e, list(sol))
            if u is None or ds > max_density:
                u, max_density = e, ds
        assert u is not None
        if cur_cost + model.cost_of_singleton(u) <= model.budget:
            sol.add(u)
            cur_cost += model.cost_of_singleton(u)
        remaining_elements.remove(u)
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    res = {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': cur_cost,
    }
    return res
