from typing import Set
from itertools import accumulate
import bisect
import numpy as np


def marginal_delta(base_set: Set[int], remaining_set: Set[int], model):
    """Delta( b | S )"""
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
    if r < len(cumsum_costs):
        c_star = cumsum_costs[r - 1]
        coefficient = (model.budget - c_star) / model.costs_obj[t[r]]
        delta += model.marginal_gain(t[r], base_set) * coefficient
    return delta


def marginal_delta_version2(base_set: Set[int], remaining_set: Set[int], ground_set: Set[int], model):
    """Cutout"""
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
        delta_marginal += model.marginal_gain(t[i], base_set) - model.cutout_marginal_gain(s[j], ground_set)  # this should be ground set
        max_delta_marginal = max(max_delta_marginal, delta_marginal)
    delta += max_delta_marginal

    # if cumsum_costs[-1] <= b, idx = len(cumsum_costs) - 1, no interpolation term
    if r < len(cumsum_costs):
        c_star = cumsum_costs[r - 1]
        coefficient = (model.budget - c_star) / model.costs_obj[t[r]]
        delta += model.marginal_gain(t[r], base_set) * coefficient
    return delta


def G_plus(x, model, remaining_set, base_set):
    """
    x: available budget 
    """
    t = list(remaining_set)
    # sort density in descending order
    t.sort(key=lambda x: model.density(x, base_set), reverse=True)
    costs = [model.costs_obj[x] for x in t]
    cumsum_costs = list(accumulate(costs, initial=None))
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
        G += model.marginal_gain(t[i], base_set)
    last_weight = (x - cumsum_costs[r1 - 1]) / model.cost_of_singleton(t[r1])
    G += last_weight * model.marginal_gain(t[r1], base_set)
    return G


def G_minus(x, model, base_set):
    """
    Difference between G_plus and G_minus
    G_plus            G_minus
    desending         ascending
    density           cutout_density
    marginal_gain     cutout_marginal_gain
    """
    s = list(base_set)
    # sort density in ascending order
    s.sort(key=lambda x: model.cutout_density(x, base_set), reverse=False)
    costs = [model.cost_of_singleton(x) for x in s]
    cumsum_costs = list(accumulate(costs, initial=None))
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
        G += model.cutout_marginal_gain(s[i], base_set)
    last_weight = (x - cumsum_costs[r1 - 1]) / model.cost_of_singleton(s[r1])
    G += last_weight * model.cutout_marginal_gain(s[r1], base_set)
    return G


def marginal_delta_version3(base_set: Set[int], remaining_set: Set[int], ground_set: Set[int], model, steps=10):
    """Cutout with knapsack cost continuous extension"""
    cs = model.cost_of_set(base_set)
    c1 = model.budget - cs
    delta = 0.
    for y in np.linspace(0, cs, num=steps):
        # note that cutout base set is ground set
        delta = max(delta, G_plus(y + c1, model, remaining_set,
                    base_set) - G_minus(y, model, ground_set))
    return delta
