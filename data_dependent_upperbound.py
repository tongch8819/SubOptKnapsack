from base_task import BaseTask

from typing import Set, List
from itertools import accumulate
import bisect
import numpy as np


def marginal_delta(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    """Delta( b | S )"""
    assert len(base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    t = list(remaining_set)
    t.sort(key=lambda x: model.density(x, base_set), reverse=True)
    costs = [model.cost_of_singleton(x) for x in t]
    # cumsum_costs[i] = sum(costs[:i+1])
    cumsum_costs = list(accumulate(costs, initial=None))
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
    check_delta = G_plus(model.budget, model, remaining_set,
                         base_set, cumsum_costs, t)
    assert abs(delta - check_delta) < 1e-6, "Inconsistency"
    return delta


def marginal_delta_version2(base_set: Set[int], remaining_set: Set[int], ground_set: Set[int], model: BaseTask):
    """Cutout"""
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    # t is the sorted remaining elements using descending density
    t = list(remaining_set)
    t.sort(key=lambda x: model.density(x, base_set), reverse=True)
    costs = [model.cost_of_singleton(x) for x in t]
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


def G_plus(x: float, model: BaseTask, remaining_set: Set[int], base_set: Set[int], cumsum_costs: List[float], elements: List[int]):
    """
    Inputs:
    - x: available budget 
    - cumsum_costs: cumsum_costs[i] = sum(costs[:i+1])
    - elements: sorted elements corresponding to cumsum_costs
    """
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


def G_minus(x: float, model: BaseTask, base_set: Set[int], cumsum_costs: List[float], elements: List[int]):
    """
    Inputs:
    - base_set: starting base set of cut out marginal gain

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


def marginal_delta_version3(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    """Cutout with knapsack cost continuous extension"""
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    cost_base_set = model.cost_of_set(base_set)
    c1 = model.budget - cost_base_set

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_density(x, base_set), reverse=False)
        costs = [model.cost_of_singleton(x) for x in s]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, s

    def outside_cumsum_costs():
        t = list(remaining_set)
        # sort density in descending order
        t.sort(key=lambda x: model.density(x, base_set), reverse=True)
        costs = [model.cost_of_singleton(x) for x in t]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, t

    # outside base set
    csc_outside, ele_outside = outside_cumsum_costs()
    csc_inside, ele_inside = inside_cumsum_costs()

    endpoints = [0.]
    tt = csc_outside[bisect.bisect_right(csc_outside, c1):bisect.bisect_right(csc_outside, model.budget)]
    endpoints += [x - c1 for x in tt]
    endpoints += csc_inside[:bisect.bisect_right(csc_inside, cost_base_set)]
    endpoints += [cost_base_set]
    endpoints.sort()

    delta = 0.
    # the y must be the end point of either G_plus or G_minus
    for y in endpoints:
        if y > cost_base_set:
            break
        # note that cutout base set is ground set
        g_plus = G_plus(y + c1, model, remaining_set,
                        base_set, csc_outside, ele_outside)
        g_minus = G_minus(y, model, set(model.ground_set), csc_inside, ele_inside)
        assert g_plus >= 0., f"G_plus({y:.2f} + {c1:.2f}) = {g_plus}"
        assert g_minus >= 0., f"G_minus({y}) = {g_minus}"
        delta = max(delta, g_plus - g_minus)
    return delta

def marginal_delta_version4(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    cost_base_set = model.cost_of_set(base_set)
    c1 = model.budget - cost_base_set

    def local_density(base_set, ele, model: BaseTask):
        if ele == -1:
            return 0
        return model.density(ele, base_set)

    def outside_cumsum_costs():
        t = list(remaining_set)
        # sort density in descending order
        t.sort(key=lambda x: model.density(x, base_set), reverse=True)
        costs = [model.cost_of_singleton(x) for x in t]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, t

    def G_over_N(base_set: Set[int], csc: List[float], ele,  val: List[float], x: float, model: BaseTask):
        idx = bisect.bisect_right(csc, x) - 1
        if idx == -1:
            return local_density(base_set, ele[0], model) * x
        else:
            return val[idx] + local_density(base_set, ele[idx], model) * (x - csc[idx])

    def f_s(base_set: Set[int], csc: List[float], ele, start: float, slice_len: float, model: BaseTask):
        start_idx = bisect.bisect_right(csc, start)
        stop_idx = bisect.bisect_left(csc, start+slice_len)

        if start_idx >= len(ele):
            return 0

        if start_idx == stop_idx:
            return local_density(base_set, ele[start_idx], model) * slice_len
        else:
            cur_idx = start_idx
            slice_1_len = csc[start_idx] - start
            val = local_density(base_set, ele[start_idx], model) * slice_1_len
            while cur_idx < stop_idx:
                val += local_density(base_set, ele[cur_idx], model) * model.cost_of_singleton(ele[cur_idx])
                cur_idx += 1
            if stop_idx < len(ele):
                slice_2_len = slice_len - slice_1_len
                val += local_density(base_set, ele[stop_idx], model) * slice_2_len

            return val

    n = len(remaining_set)

    slice_length = c1 / n

    csc_outside, ele_outside = outside_cumsum_costs()

    csc_outside.append(c1)

    val_outside = [model.objective([ele]) for ele in ele_outside]

    val_outside.append(0)

    val_outside = list(accumulate(val_outside, initial=None))

    ele_outside.append(-1)

    endpoints = csc_outside[:bisect.bisect_right(csc_outside, c1)]
    endpoints.sort()

    delta = 0.
    i_j_right_idx = 0

    for j in range(0, n):
        v_j = 0.

        while True:
            left = G_over_N(base_set,
                            csc_outside,
                            ele_outside,
                            val_outside,
                            csc_outside[i_j_right_idx], model) - delta

            right = f_s(base_set, csc_outside, ele_outside, csc_outside[i_j_right_idx] - slice_length, slice_length, model)

            if left - right >= 0:
                left_k = local_density(base_set, ele_outside[i_j_right_idx], model)
                right_k = 0.
                if i_j_right_idx == 0:
                    right_k = right / csc_outside[0]
                else:
                    prev_right = f_s(base_set, csc_outside, ele_outside, csc_outside[i_j_right_idx-1] - slice_length, slice_length, model)
                    right_k = (right - prev_right) / (csc_outside[i_j_right_idx] - csc_outside[i_j_right_idx-1])

                i_star = csc_outside[i_j_right_idx] - (left - right)/(left_k - right_k)

                v_j = f_s(base_set, csc_outside, ele_outside, i_star - slice_length, slice_length, model)
                break
            else:
                i_j_right_idx += 1

        delta += v_j
    return delta