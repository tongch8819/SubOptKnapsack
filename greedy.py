def greedy(model):
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    while len(remaining_elements):
        # 
        u, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            # mg = marginal_gain(objective, e, sol)
            # density = (mg * 100) / (costs_obj[e] * 100)  # cost may be too small
            ds = model.density(e, sol)
            if u is None or ds > max_density:
                u, max_density = e, ds
        #TODO: filter out violating elements
        assert u is not None
        if cur_cost + model.costs_obj[u] <= model.budget:
            sol.add(u)
            cur_cost += model.costs_obj[u]
        remaining_elements.remove(u)
    res = {
        'S' : sol,
        'f(S)' : model.objective(sol),
        'c(S)' : cur_cost,
    }
    return res

