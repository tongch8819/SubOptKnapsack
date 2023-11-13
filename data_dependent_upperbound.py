from typing import Set
from itertools import accumulate
import bisect
import numpy as np


def marginal_delta(base_set: Set[int], remaining_set: Set[int], model):
    """Delta( b | S )"""
    assert len(base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    t = list(remaining_set)
    t.sort(key=lambda x: model.density(x, base_set), reverse=True)
    costs = [model.costs_obj[x] for x in t]
    cumsum_costs = list(accumulate(costs, initial=None))
    # cumsum_costs[i] = sum(costs[:i])  exclusive
    # idx = bisect.bisect_left(cumsum_costs, b)
    # cumsum_costs[:idx]: x < b
    # cumsum_costs[idx:]: x >= b
    idx = bisect.bisect_right(cumsum_costs, model.budget)
    # cumsum_costs[:idx]: x <= b
    # cumsum_costs[idx:]: x > b
    r = idx

    delta = 0.
    for i in range(r):
        # t[i] is a single element
        delta += model.marginal_gain(t[i], base_set)

    # if cumsum_costs[-1] <= b, idx = len(cumsum_costs) - 1, no interpolation term
    if r >= 1 and r < len(cumsum_costs):
        c_star = cumsum_costs[r - 1]
        coefficient = (model.budget - c_star) / model.costs_obj[t[r]]
        delta += model.marginal_gain(t[r], base_set) * coefficient

    # G_plus(x, model, remaining_set, base_set, cumsum_costs, elements):
    check_delta = G_plus(model.budget, model, remaining_set, base_set, cumsum_costs, t)
    assert abs(delta - check_delta) < 1e-6, "Inconsistency"
    return delta


def marginal_delta_version2(base_set: Set[int], remaining_set: Set[int], ground_set: Set[int], model):
    """Cutout"""
    assert len(base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    # t is the sorted remaining elements using descending density
    t = list(remaining_set)
    t.sort(key=lambda x: model.density(x, base_set), reverse=True)
    costs = [model.costs_obj[x] for x in t]
    cumsum_costs = list(accumulate(costs, initial=None))
    # cumsum_costs[i] = sum(costs[:i])  exclusive
    # idx = bisect.bisect_left(cumsum_costs, b)
    # cumsum_costs[:idx]: x < b
    # cumsum_costs[idx:]: x >= b
    r1 = bisect.bisect_right(
        cumsum_costs, model.budget - model.cost_of_set(base_set))
    r = bisect.bisect_right(cumsum_costs, model.budget)
    # cumsum_costs[:idx]: x <= b
    # cumsum_costs[idx:]: x > b

    # t is the sorted remaining elements using descending density
    s = list(base_set)
    s.sort(key=lambda x: model.cutout_marginal_gain(x, base_set), reverse=False)

    delta = 0.
    for i in range(r1):
        # t[i] is a single element
        delta += model.marginal_gain(t[i], base_set)

    # find max
    delta_marginal, max_delta_marginal = 0., -1
    for i in range(r1, r):
        # t[i] is a single element
        j = i - r1
        if j == len(s):
            break
        delta_marginal += model.marginal_gain(t[i], base_set) - model.cutout_marginal_gain(
            s[j], ground_set)  # this should be ground set
        max_delta_marginal = max(max_delta_marginal, delta_marginal)
    delta += max_delta_marginal

    # if cumsum_costs[-1] <= b, idx = len(cumsum_costs) - 1, no interpolation term
    if r < len(cumsum_costs):
        c_star = cumsum_costs[r - 1]
        coefficient = (model.budget - c_star) / model.costs_obj[t[r]]
        delta += model.marginal_gain(t[r], base_set) * coefficient
    return delta


def G_plus(x, model, remaining_set, base_set, cumsum_costs, elements):
    """
    x: available budget 
    """
    # cumsum_costs[i] = sum(costs[:i])  exclusive
    # idx = bisect.bisect_left(cumsum_costs, b)
    # cumsum_costs[:idx]: x < b
    # cumsum_costs[idx:]: x >= b
    r1 = bisect.bisect_right(cumsum_costs, x)
    # cumsum_costs[:idx]: x <= b
    # cumsum_costs[idx:]: x > b
    G = 0.
    for i in range(r1):
        # t[i] is a single element
        G += model.marginal_gain(elements[i], base_set)
    if r1 >= 1 and r1 < len(cumsum_costs):
        last_weight = (x - cumsum_costs[r1 - 1]) / \
            model.cost_of_singleton(elements[r1])
        G += last_weight * model.marginal_gain(elements[r1], base_set)
    return G


def G_minus(x, model, base_set, cumsum_costs, elements):
    """
    Difference between G_plus and G_minus
    G_plus            G_minus
    desending         ascending
    density           cutout_density
    marginal_gain     cutout_marginal_gain
    """
    # cumsum_costs[i] = sum(costs[:i])  exclusive
    # idx = bisect.bisect_left(cumsum_costs, b)
    # cumsum_costs[:idx]: x < b
    # cumsum_costs[idx:]: x >= b
    r1 = bisect.bisect_right(cumsum_costs, x)
    # cumsum_costs[:idx]: x <= b
    # cumsum_costs[idx:]: x > b
    G = 0.
    for i in range(r1):
        # t[i] is a single element
        G += model.cutout_marginal_gain(elements[i], base_set)
    if r1 >= 1 and r1 < len(cumsum_costs):
        last_weight = (x - cumsum_costs[r1 - 1]) / \
            model.cost_of_singleton(elements[r1])
        assert last_weight >= 0., f"last weight: {last_weight}, x: {x}, cumsum: {cumsum_costs}, r: {r1}"
        G += last_weight * model.cutout_marginal_gain(elements[r1], base_set)
    return G


def marginal_delta_version3(base_set: Set[int], remaining_set: Set[int], ground_set: Set[int], model):
    """Cutout with knapsack cost continuous extension"""
    assert len(base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    cs = model.cost_of_set(base_set)
    c1 = model.budget - cs

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order
        s.sort(key=lambda x: model.cutout_density(x, base_set), reverse=False)
        costs = [model.cost_of_singleton(x) for x in s]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, s

    def outside_cumsum_costs():
        t = list(remaining_set)
        # sort density in descending order
        t.sort(key=lambda x: model.density(x, base_set), reverse=True)
        costs = [model.costs_obj[x] for x in t]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, t

    csc_outside, ele_outside = outside_cumsum_costs()
    csc_inside, ele_inside = inside_cumsum_costs()

    endpoints = csc_outside[:bisect.bisect_right(csc_outside, model.budget)]
    endpoints += csc_inside[:bisect.bisect_right(csc_inside, cs)]
    endpoints.sort()

    delta = 0.
    # the y must be the end point of either G_plus or G_minus
    for y in endpoints:
        # note that cutout base set is ground set
        g_plus = G_plus(y + c1, model, remaining_set, base_set, csc_outside, ele_outside)
        g_minus = G_minus(y, model, ground_set, csc_inside, ele_inside)
        assert g_plus >= 0., f"G_plus({y:.2f} + {c1:.2f}) = {g_plus}"
        assert g_minus >= 0., f"G_minus({y}) = {g_minus}"
        delta = max(delta, g_plus - g_minus)
    return delta
