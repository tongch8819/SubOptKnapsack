from constraint import ProblemInstance


def greedy_cardinality(model: ProblemInstance):
    sol = set()
    remaining_elements = set(model.objective.ground_set)
    k = model.constraint.size

    ratio_lst = []
    factor_lst = []

    for i in range(k):
        u = None
        for e in remaining_elements:
            mg = model.objective.marginal_gain(e, list(sol))


            if u is None or mg > max_marginal_gain:
                u, max_marginal_gain = e, mg
        assert u is not None
        # add optimal element into solution set
        sol.add(u)
        remaining_elements.remove(u)

        # add statistics'
        singleton_value = model.objective.eval([u])
        ratio = max_marginal_gain / singleton_value
        ratio_lst.append(ratio)

        l = len(sol)
        if abs(1 - ratio) < 1e-4:
            t = 0
        else:
            t = l / ((1 - ratio) * k + 1e-4) 
        factor_lst.append(t)

    res = {
        'S': sol,
        'f(S)': model.objective.eval(sol),
        'ratio list': ratio_lst,
        'factor list': factor_lst,
    }
    return res
