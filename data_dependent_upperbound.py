import time

from base_task import BaseTask

from typing import Set, List
from itertools import accumulate
import bisect
import numpy as np


def marginal_delta(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    """Delta( b | S )"""
    assert len(base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    if len(remaining_set) == 0:
        return 0

    t = list(remaining_set)
    t.sort(key=lambda x: model.density(x, base_set), reverse=True)
    costs = [model.cost_of_singleton(x) for x in t]
    # # cumsum_costs[i] = sum(costs[:i+1])
    cumsum_costs = list(accumulate(costs, initial=None))
    # # idx = bisect.bisect_left(cumsum_costs, b)
    # # cumsum_costs[:idx]: x < b
    # # cumsum_costs[idx:]: x >= b
    # idx = bisect.bisect_right(cumsum_costs, model.budget)
    # # cumsum_costs[:idx]: x <= b
    # # cumsum_costs[idx:]: x > b
    # r = idx
    #
    # delta = 0.
    # for i in range(r):
    #     # t[i] is a single element
    #     delta += model.marginal_gain(t[i], base_set)
    #
    # # if cumsum_costs[-1] <= b, idx = len(cumsum_costs) - 1, no interpolation term
    # if r >= 1 and r < len(cumsum_costs):
    #     c_star = cumsum_costs[r - 1]
    #     coefficient = (model.budget - c_star) / model.costs_obj[t[r]]
    #     delta += model.marginal_gain(t[r], base_set) * coefficient

    # G_plus(x, model, remaining_set, base_set, cumsum_costs, elements):
    delta = G_plus(model.budget, model, remaining_set,
                         base_set, cumsum_costs, t)
    # assert abs(delta - check_delta) < 1e-6, "Inconsistency"
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


def G_plus(x: float, model: BaseTask, remaining_set: Set[int], base_set: Set[int], cumsum_costs: List[float],
           elements: List[int]):
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
    if r1 == 0:
        return x * model.marginal_gain(elements[0], base_set) / model.cost_of_singleton(elements[0])
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
    if x == 0:
        return 0
    # cumsum_costs[i] = sum(costs[:i])  exclusive
    # idx = bisect.bisect_left(cumsum_costs, b)
    # cumsum_costs[:idx]: x < b
    # cumsum_costs[idx:]: x >= b
    r1 = bisect.bisect_right(cumsum_costs, x)
    # cumsum_costs[:idx]: x <= b
    # cumsum_costs[idx:]: x > b
    G = 0.
    if r1 == 0:
        return x * model.cutout_density(elements[0], base_set)
    for i in range(r1):
        # t[i] is a single element
        G += model.cutout_marginal_gain(elements[i])

    if r1 >= 1 and r1 < len(cumsum_costs):
        last_weight = x - cumsum_costs[r1 - 1]
        assert last_weight >= 0., f"last weight: {last_weight}, x: {x}, cumsum: {cumsum_costs}, r: {r1}"
        G += last_weight * model.cutout_density(elements[r1], base_set)
    return G


def marginal_delta_version3(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    """Cutout with knapsack cost continuous extension"""
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    cost_base_set = model.cost_of_set(base_set)
    c1 = model.budget - cost_base_set

    if len(remaining_set) == 0:
        return 0

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

    # print(f"ele_outside:{ele_outside[:10]},ele_inside:{ele_inside[:10]}, base:{base_set}")

    endpoints = [0.]
    tt = csc_outside[bisect.bisect_right(csc_outside, c1):bisect.bisect_right(csc_outside, model.budget)]
    endpoints += [x - c1 for x in tt]
    endpoints += csc_inside[:bisect.bisect_right(csc_inside, cost_base_set)]
    endpoints += [cost_base_set]
    endpoints.sort()

    # return G_plus(model.budget, model, remaining_set, base_set, csc_outside, ele_outside)

    delta = 0.
    # the y must be the end point of either G_plus or G_minus
    for y in endpoints:
        if y > cost_base_set:
            break
        # note that cutout base set is ground set
        g_plus = G_plus(y + c1, model, remaining_set,
                        base_set, csc_outside, ele_outside)
        g_minus = G_minus(y, model, set(model.ground_set), csc_inside, ele_inside)

        # print(f"g plus is:{g_plus}, y is:{y}, y+c1:{y+c1}, g minus is:{g_minus}")

        assert g_plus >= 0., f"G_plus({y:.2f} + {c1:.2f}) = {g_plus}"
        assert g_minus >= 0., f"G_minus({y}) = {g_minus}"
        delta = max(delta, g_plus - g_minus)

    # print(f"delta is:{delta}, max is:{G_plus(model.budget, model, remaining_set, base_set, csc_outside, ele_outside)}")
    return delta


def marginal_delta_version4(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    if len(remaining_set) == 0:
        return 0

    c1 = model.budget

    base_set_value = model.objective(base_set)

    n = 5

    eps = c1 / n

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_density(x, base_set), reverse=False)
        costs = [model.cost_of_singleton(x) for x in s]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, s

    def local_density(f, base, ele):
        if ele == -1:
            return 0
        return (f(base | {ele}) - f(base)) / model.cost_of_singleton(ele)

    total_utility = model.objective(model.ground_set)

    def f_over_base(s):
        return model.objective(base_set | s) - base_set_value

    def f_minus(s):
        return total_utility - model.objective(set(model.ground_set) - set(s))

    def method3(f, b):
        def outside_cumsum_costs(f, b):
            t = list(remaining_set)
            t.append(-1)
            # sort density in descending order
            t.sort(key=lambda x: local_density(f, set(), x), reverse=True)

            costs = [model.cost_of_singleton(x) for x in t]
            costs.append(b)

            cumsum_costs = list(accumulate(costs, initial=None))
            return cumsum_costs, t

        csc_outside, ele_outside = outside_cumsum_costs(f, b)

        endpoints_enter = [0] + csc_outside

        endpoints_leave = [x + eps for x in endpoints_enter]

        endpoints = list(set(endpoints_leave + endpoints_enter))

        endpoints.sort()

        def G_over_N(x: float):
            idx = bisect.bisect_right(csc_outside, x) - 1
            if idx == -1:
                return local_density(f, set(), ele_outside[0]) * x
            else:
                return f(set(ele_outside[:idx])) \
                       + local_density(f, set(ele_outside[:idx]), ele_outside[idx]) * (x - csc_outside[idx])

        def f_s(start: float, eps: float):
            start_idx = bisect.bisect_right(csc_outside, start)
            stop_idx = bisect.bisect_left(csc_outside, start + eps)

            if start_idx >= len(ele_outside):
                return 0

            if start_idx == stop_idx:
                return local_density(f, set(), ele_outside[start_idx]) * eps
            else:
                slice_1_len = csc_outside[start_idx] - start
                val = local_density(f, set(), ele_outside[start_idx]) * slice_1_len
                eps -= slice_1_len

                cur_idx = start_idx + 1

                while eps > 0:
                    cur_ele_cost = model.cost_of_singleton(ele_outside[cur_idx])
                    if cur_ele_cost <= eps:
                        val += local_density(f, set(), ele_outside[cur_idx]) * cur_ele_cost

                        eps -= cur_ele_cost
                        cur_idx += 1

                    else:
                        val += local_density(f, set(), ele_outside[cur_idx]) * eps
                        break

                return val

        delta = 0.
        endpoint_idx = 0
        prev_i_star = 0

        result = []

        budget_consumed = 0

        while budget_consumed < b:
            c_eps = min(eps, b - budget_consumed)

            v_j = 0.

            while True:
                ele_idx = bisect.bisect_left(csc_outside, endpoints[endpoint_idx])

                if endpoints[endpoint_idx] - prev_i_star >= c_eps:
                    left = G_over_N(endpoints[endpoint_idx]) - delta

                    right = f_s(endpoints[endpoint_idx] - c_eps, c_eps)

                    if left - right >= 0:
                        if ele_idx == 0:
                            left_k = local_density(f, set(), ele_outside[ele_idx])
                        else:
                            left_k = local_density(f, set(ele_outside[:ele_idx - 1]), ele_outside[ele_idx])

                        if endpoint_idx == 0:
                            right_k = right / endpoints[0]
                        else:
                            prev_right = f_s(endpoints[endpoint_idx - 1] - c_eps, c_eps)
                            right_k = (right - prev_right) / (csc_outside[endpoint_idx] - csc_outside[endpoint_idx - 1])

                        i_star = max(endpoints[endpoint_idx] - (left - right) / (left_k - right_k),
                                     prev_i_star + c_eps)

                        v_j = f_s(i_star - c_eps, c_eps)

                        prev_i_star = i_star

                        break
                    else:
                        endpoint_idx += 1
                else:
                    endpoint_idx += 1

            delta += v_j
            budget_consumed += c_eps

            result.append((budget_consumed, delta))

        return result

    def method4(f, b):
        def outside_cumsum_costs(f, b):
            t = list(remaining_set)
            t.append(-1)
            # sort density in descending order
            t.sort(key=lambda x: local_density(f, set(), x), reverse=False)

            costs = [model.cost_of_singleton(x) for x in t]
            costs.append(b)

            cumsum_costs = list(accumulate(costs, initial=None))
            return cumsum_costs, t

        csc_outside, ele_outside = outside_cumsum_costs(f, b)

        endpoints_enter = [0] + csc_outside

        endpoints_leave = [x + eps for x in endpoints_enter]

        endpoints = list(set(endpoints_leave + endpoints_enter))

        endpoints.sort()

        def G_over_N(x: float):
            idx = bisect.bisect_right(csc_outside, x) - 1
            if idx == -1:
                return local_density(f, {}, ele_outside[0]) * x
            else:
                return f(set(ele_outside[:idx])) \
                       + local_density(f, set(ele_outside[:idx]), ele_outside[idx]) * (x - csc_outside[idx])

        def f_s(start: float, eps: float):
            start_idx = bisect.bisect_right(csc_outside, start)
            stop_idx = bisect.bisect_left(csc_outside, start + eps)

            if start_idx >= len(ele_outside):
                return 0

            if start_idx == stop_idx:
                return local_density(f, set(), ele_outside[start_idx]) * eps
            else:
                slice_1_len = csc_outside[start_idx] - start
                val = local_density(f, set(), ele_outside[start_idx]) * slice_1_len
                eps -= slice_1_len

                cur_idx = start_idx + 1

                while eps > 0:
                    cur_ele_cost = model.cost_of_singleton(ele_outside[cur_idx])
                    if cur_ele_cost <= eps:
                        val += local_density(f, set(), ele_outside[cur_idx]) * cur_ele_cost

                        eps -= cur_ele_cost
                        cur_idx += 1

                    else:
                        val += local_density(f, set(), ele_outside[cur_idx]) * eps
                        break

                return val

        delta = 0.
        endpoint_idx = 0
        prev_i_star = 0

        result = []

        budget_consumed = 0

        while budget_consumed < b:
            c_eps = min(eps, b - budget_consumed)

            v_j = 0.

            while True:
                ele_idx = bisect.bisect_left(csc_outside, endpoints[endpoint_idx])

                if endpoints[endpoint_idx] - prev_i_star >= c_eps:
                    left = G_over_N(endpoints[endpoint_idx]) - delta

                    right = f_s(endpoints[endpoint_idx] - c_eps, c_eps)

                    if left - right >= 0:
                        if ele_idx == 0:
                            left_k = local_density(f, {}, ele_outside[ele_idx])
                        else:
                            left_k = local_density(f, set(ele_outside[:ele_idx - 1]), ele_outside[ele_idx])

                        if endpoint_idx == 0:
                            right_k = right / endpoints[0]
                        else:
                            prev_right = f_s(endpoints[endpoint_idx - 1] - c_eps, c_eps)
                            right_k = (right - prev_right) / (csc_outside[endpoint_idx] - csc_outside[endpoint_idx - 1])

                        i_star = max(endpoints[endpoint_idx] - (left - right) / (left_k - right_k),
                                     prev_i_star + c_eps)

                        v_j = f_s(i_star - c_eps, c_eps)

                        prev_i_star = i_star

                        break
                    else:
                        endpoint_idx += 1
                else:
                    endpoint_idx += 1

            delta += v_j
            budget_consumed += c_eps

            result.append((budget_consumed, delta))

        return result

    ub = 0

    M_plus_res = method3(f_over_base, model.budget)

    M_plus_budget = [x[0] for x in M_plus_res]
    M_plus_gain = [x[1] for x in M_plus_res]

    def M_plus(x):
        idx = bisect.bisect_left(M_plus_budget, x) - 1
        if idx < 0:
            return x * M_plus_gain[0] / M_plus_budget[0]
        else:
            return M_plus_gain[idx] + (x - M_plus_budget[idx]) * (M_plus_gain[idx + 1] - M_plus_gain[idx]) / (
                        M_plus_budget[idx + 1] - M_plus_budget[idx])


    endpoints_plus = [x[0] for x in M_plus_res]

    minimal_budget = model.budget - model.cost_of_set(base_set)
    cost_baseset = model.cost_of_set(base_set)

    endpoints = [0.]

    csc_inside, ele_inside = inside_cumsum_costs()
    endpoints_minus = csc_inside[:bisect.bisect_right(csc_inside, cost_baseset)]
    endpoints_minus = [x + minimal_budget for x in endpoints_minus]

    endpoints += list(set(endpoints_plus) | set(endpoints_minus))

    endpoints.append(model.budget)

    endpoints = list(set(endpoints))

    endpoints.sort()

    for i in range(0, len(endpoints)):
        if endpoints[i] >= minimal_budget:
            g_minus = G_minus(endpoints[i] - minimal_budget, model, set(model.ground_set), csc_inside, ele_inside)
            t_ub = M_plus(endpoints[i]) - g_minus
            #print(f"t ub is:{t_ub}, g minus is:{g_minus}, y is:{endpoints[i] - model.budget + base_budget},t:{model.budget-base_budget},b:{base_budget}")
            if t_ub > ub:
                ub = t_ub

    return ub


def marginal_delta_version5(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    if len(remaining_set) == 0:
        return 0

    base_set_value = model.objective(base_set)

    eps = min([model.cost_of_singleton(ele) for ele in remaining_set])/4

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_density(x, base_set), reverse=False)
        costs = [model.cost_of_singleton(x) for x in s]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, s

    def local_f(S):
        S = list(S)
        while S.count(-1) > 0:
            S.remove(-1)
        return model.objective(S)

    def local_c(S):
        S = list(S)
        ept = S.count(-1) * eps
        while S.count(-1) > 0:
            S.remove(-1)
        return ept + model.cost_of_set(S)

    def local_c_singelton(ele: int):
        if ele == -1:
            return eps
        return model.cost_of_singleton(ele)

    def local_density(f, base, ele):
        return (f(base | {ele}) - f(base)) / model.cost_of_singleton(ele)

    total_utility = model.objective(model.ground_set)

    def f_over_base(s):
        return local_f(base_set | s) - base_set_value

    def method3(f, b):
        def outside_cumsum_costs(f):
            eles = list(remaining_set)
            # sort density in descending order
            eles.sort(key=lambda x: local_density(f, set(), x), reverse=True)
            costs = [model.cost_of_singleton(x) for x in eles]

            cur_idx = 1
            for i in range(0, len(eles)):
                eles.insert(cur_idx, -1)
                cur_idx = cur_idx + 2

            cur_idx = 1
            for i in range(0, len(costs)):
                costs.insert(cur_idx, eps)
                cur_idx = cur_idx + 2

            cumsum_costs = list(accumulate(costs, initial=None))
            return cumsum_costs, eles

        csc_outside, ele_outside = outside_cumsum_costs(f)

        """
        sum = 0
        for i in range(0, min(100, len(ele_outside))):
            sum += f({ele_outside[i]})
            if ele_outside[i] >= 0:
                prev = 0.
                if i > 0:
                    prev = f(set(ele_outside[:i-1]))
                print(f"f(A):{f(set(ele_outside[:i+1]))},sum:{sum},sin:{f({ele_outside[i]})},sub:{f(set(ele_outside[:i+1])) - prev},ele:{ele_outside[i]}")
        """

        def G_over_N(x: float):
            idx = bisect.bisect_right(csc_outside, x) - 1
            if idx == -1:
                return local_density(f, set(), ele_outside[0]) * x
            else:
                return f(set(ele_outside[:idx])) \
                       + local_density(f, set(ele_outside[:idx]), ele_outside[idx]) * (x - csc_outside[idx])

        def f_s(start: float):
            ks = bisect.bisect_right(csc_outside, start)
            if ks >= len(ele_outside):
                return 0

            slice_len = min(csc_outside[ks] - start, eps)
            val = local_density(f, set(), ele_outside[ks]) * slice_len

            return val

        delta = 0.

        start_from_new_ele = False

        ele_idx = 0
        prev_i_star = 0

        result = []

        budget_consumed = 0

        # print(f"new turn, csc:{csc_outside[:10]}")

        while budget_consumed < b - eps*0.01:
            v_j = 0.

            rbd = 0.

            if budget_consumed + eps <= b:
                # first, check the start point
                if start_from_new_ele:
                    ele_idx = ele_idx + 2
                    if ele_idx >= len(csc_outside) - 1:
                        break
                    stp_idx = csc_outside[ele_idx-1] + eps
                else:
                    stp_idx = prev_i_star + eps

                if ele_idx >= len(csc_outside) - 1:
                    break

                stp_left = f_s(stp_idx - eps)
                stp_right = G_over_N(stp_idx) - delta

                stp_value = min(stp_left, stp_right)
                stp_for_cur_ele = csc_outside[ele_idx]

                check_subsequent_elements = False

                rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)

                if stp_left <= stp_right and stp_idx < stp_for_cur_ele:
                    v_j = stp_value
                    prev_i_star = stp_idx
                    start_from_new_ele = False
                    # print(f"stop at -1, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx}, c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
                elif stp_left <= stp_right and stp_idx >= stp_for_cur_ele:
                    v_j = stp_value
                    prev_i_star = stp_idx
                    start_from_new_ele = True
                    check_subsequent_elements = True
                    # print(f"update at -1, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
                else:
                    # second, check the point where f sin = G N in this element
                    cur_marginal_density = local_density(f, set(ele_outside[:ele_idx]), ele_outside[ele_idx])
                    t_delta = (stp_left - stp_right) / cur_marginal_density
                    if stp_idx + t_delta <= stp_for_cur_ele:
                        # subsequent elements cannot bring higher value
                        v_j = stp_right + cur_marginal_density * t_delta
                        stp_idx = stp_idx + t_delta
                        prev_i_star = stp_idx
                        start_from_new_ele = False
                        rbd = eps
                        # print(f"update at 0, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
                        # print(f"stop at 0, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
                    else:
                        stp_for_cur_ele_left = f_s(stp_for_cur_ele - eps)
                        stp_for_cur_ele_right = G_over_N(stp_for_cur_ele) - delta

                        cur_singleton_density = local_density(f, set(), ele_outside[ele_idx])
                        t_delta = (stp_for_cur_ele_left - stp_for_cur_ele_right)/cur_singleton_density
                        t_vj = stp_for_cur_ele_left - t_delta * cur_singleton_density
                        if t_vj >= v_j:
                            v_j = t_vj
                            stp_idx = stp_for_cur_ele + t_delta
                            prev_i_star = stp_idx
                            start_from_new_ele = True
                            rbd = csc_outside[ele_idx] - (stp_idx - eps)
                            # print(f"update at 1.5, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")

                        check_subsequent_elements = True

                if check_subsequent_elements:
                    # third, check the start point and the point where f sin = G N in subsequent elements until
                    # the element whose start point satisfying f sin <= G N
                    while True:
                        # start point
                        ele_idx = ele_idx + 2
                        if ele_idx >= len(csc_outside) - 1:
                            # print(f"stop at 1, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]})/1}")
                            break

                        t_stp_idx = csc_outside[ele_idx-1] + eps

                        t_stp_left = f_s(t_stp_idx - eps)
                        t_stp_right = G_over_N(t_stp_idx) - delta

                        t_v_j = min(t_stp_left, t_stp_right)
                        if t_v_j >= v_j:
                            v_j = t_v_j
                            stp_idx = t_stp_idx
                            prev_i_star = stp_idx
                            rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                            if stp_idx + eps >= csc_outside[ele_idx]:
                                start_from_new_ele = True
                            else:
                                start_from_new_ele = False
                            #print(f"update at 2, ele {ele_idx}, left {t_stp_left}, right {t_stp_right}, v j {v_j}, idx:{t_stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]})/1}")

                        if t_stp_left <= t_stp_right:
                            #print(f"stop at 2, ele {ele_idx}, left {t_stp_left}, right {t_stp_right}, v j {v_j}, idx:{t_stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]})/1}")
                            break

                        # balance point
                        cur_marginal_density = local_density(f, set(ele_outside[:ele_idx]), ele_outside[ele_idx])
                        stp_for_cur_ele = csc_outside[ele_idx]

                        t_delta = (t_stp_left - t_stp_right) / cur_marginal_density
                        if t_stp_idx + t_delta <= stp_for_cur_ele:
                            # subsequent elements cannot bring higher value
                            t_v_j = t_stp_right + cur_marginal_density * t_delta
                            if t_v_j >= v_j:
                                v_j = t_v_j
                                stp_idx = t_stp_idx
                                prev_i_star = stp_idx
                                rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                                if stp_idx + eps >= csc_outside[ele_idx]:
                                    start_from_new_ele = True
                                else:
                                    start_from_new_ele = False
                            break
                        else:
                            stp_for_cur_ele_left = f_s(stp_for_cur_ele - eps)
                            stp_for_cur_ele_right = G_over_N(stp_for_cur_ele) - delta

                            cur_singleton_density = local_density(f, set(), ele_outside[ele_idx])
                            t_delta = (stp_for_cur_ele_left - stp_for_cur_ele_right) / cur_singleton_density
                            t_vj = stp_for_cur_ele_left - t_delta * cur_singleton_density
                            if t_vj >= v_j:
                                v_j = t_vj
                                stp_idx = stp_for_cur_ele + t_delta
                                rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                                prev_i_star = stp_idx
                                start_from_new_ele = True
                                #print(f"update at 4, ele {ele_idx}, left {t_stp_left}, right {t_stp_right}, v j {v_j}, idx:{t_stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
            else:
                # start point
                # check the zero point of each element
                # if left > right , check balance point, update forward to next element
                # else update, terminate
                c_eps = b - budget_consumed
                if ele_idx >= len(csc_outside) - 1:
                    break
                # first, check the start point
                if start_from_new_ele:
                    ele_idx = ele_idx + 2
                    if ele_idx >= len(csc_outside) - 1:
                        break
                    stp_idx = csc_outside[ele_idx] - c_eps + eps
                else:
                    stp_idx = max(prev_i_star + eps, csc_outside[ele_idx] - c_eps + eps)

                stp_left = f_s(stp_idx - eps)
                stp_right = G_over_N(stp_idx) - delta

                stp_value = min(stp_left, stp_right)
                stp_for_cur_ele = csc_outside[ele_idx]

                rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)

                if stp_left <= stp_right:
                    v_j = stp_value
                    prev_i_star = stp_idx
                else:
                    # check the point where f sin = G N in this element
                    stp_for_cur_ele_left = f_s(stp_for_cur_ele - eps)
                    stp_for_cur_ele_right = G_over_N(stp_for_cur_ele) - delta

                    cur_singleton_density = local_density(f, set(), ele_outside[ele_idx])
                    t_delta = (stp_for_cur_ele_left - stp_for_cur_ele_right)/cur_singleton_density
                    t_vj = stp_for_cur_ele_left - t_delta * cur_singleton_density
                    if t_vj >= v_j:
                        v_j = t_vj
                        stp_idx = stp_for_cur_ele + t_delta
                        prev_i_star = stp_idx
                        rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)

                while True:
                    ele_idx = ele_idx + 2
                    if ele_idx >= len(csc_outside) - 1:
                        break

                    t_stp_idx = csc_outside[ele_idx] - c_eps + eps
                    t_stp_left = f_s(t_stp_idx - eps)
                    t_stp_right = G_over_N(t_stp_idx) - delta

                    t_v_j = min(t_stp_left, t_stp_right)
                    if t_v_j >= v_j:
                        v_j = t_v_j
                        stp_idx = t_stp_idx
                        prev_i_star = stp_idx
                        rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)

                    if t_stp_left <= t_stp_right:
                        break

                    # balance point
                    stp_for_cur_ele_left = f_s(stp_for_cur_ele - eps)
                    stp_for_cur_ele_right = G_over_N(stp_for_cur_ele) - delta

                    cur_singleton_density = local_density(f, set(), ele_outside[ele_idx])
                    t_delta = (stp_for_cur_ele_left - stp_for_cur_ele_right) / cur_singleton_density
                    t_vj = stp_for_cur_ele_left - t_delta * cur_singleton_density
                    if t_vj >= v_j:
                        v_j = t_vj
                        stp_idx = stp_for_cur_ele + t_delta
                        prev_i_star = stp_idx
                        rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)

            delta += v_j
            budget_consumed += rbd

            #print(f"cur bd:{budget_consumed},rbd:{rbd},  vj:{v_j}, delta:{delta}, i star:{prev_i_star}, ele {ele_idx}")

            result.append((budget_consumed, delta))

        #print(f"Method 3 completes with delta:{delta}")

        return result

    ub = 0

    M_plus_res = method3(f_over_base, model.budget)

    M_plus_budget = [x[0] for x in M_plus_res]
    M_plus_gain = [x[1] for x in M_plus_res]

    #return max(M_plus_gain)

    def M_plus(x):
        idx = bisect.bisect_left(M_plus_budget, x) - 1
        if idx < 0:
            return x * M_plus_gain[0] / M_plus_budget[0]
        elif idx >= len(M_plus_gain)-1:
            return M_plus_gain[len(M_plus_gain)-1]
        else:
            return M_plus_gain[idx] + (x - M_plus_budget[idx]) * (M_plus_gain[idx + 1] - M_plus_gain[idx]) / (
                        M_plus_budget[idx + 1] - M_plus_budget[idx])


    endpoints_plus = [x[0] for x in M_plus_res]

    minimal_budget = model.budget - model.cost_of_set(base_set)
    cost_baseset = model.cost_of_set(base_set)

    endpoints = [0.]

    csc_inside, ele_inside = inside_cumsum_costs()
    endpoints_minus = csc_inside[:bisect.bisect_right(csc_inside, cost_baseset)]
    endpoints_minus = [x + minimal_budget for x in endpoints_minus]

    endpoints += list(set(endpoints_plus) | set(endpoints_minus))

    endpoints.append(model.budget)

    endpoints = list(set(endpoints))

    endpoints.sort()

    for i in range(0, len(endpoints)):
        if endpoints[i] >= minimal_budget:
            g_minus = G_minus(endpoints[i] - minimal_budget, model, set(model.ground_set), csc_inside, ele_inside)
            t_ub = M_plus(endpoints[i]) - g_minus
            # print(f"t ub is:{t_ub}, g minus is:{g_minus}, y is:{endpoints[i] - model.budget + base_budget},t:{model.budget-base_budget},b:{base_budget}")
            if t_ub > ub:
                ub = t_ub

    # print(f"delta is:{ub}, max budget:{max(M_plus_budget)}")

    return ub #max(M_plus_gain)

def marginal_delta_version5_p(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    if len(remaining_set) == 0:
        return 0

    base_set_value = model.objective(base_set)

    eps = min([model.cost_of_singleton(ele) for ele in remaining_set])/2

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_density(x, base_set), reverse=False)
        costs = [model.cost_of_singleton(x) for x in s]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, s

    def local_f(S):
        S = list(S)
        while S.count(-1) > 0:
            S.remove(-1)
        return model.objective(S)

    def local_c(S):
        S = list(S)
        ept = S.count(-1) * eps
        while S.count(-1) > 0:
            S.remove(-1)
        return ept + model.cost_of_set(S)

    def local_c_singelton(ele: int):
        if ele == -1:
            return eps
        return model.cost_of_singleton(ele)

    def local_density(f, base, ele):
        return (f(base | {ele}) - f(base)) / model.cost_of_singleton(ele)

    total_utility = model.objective(model.ground_set)

    def f_over_base(s):
        return local_f(base_set | s) - base_set_value

    def method3(f, b):
        def outside_cumsum_costs(f):
            eles = list(remaining_set)
            # sort density in descending order
            eles.sort(key=lambda x: local_density(f, set(), x), reverse=True)
            costs = [model.cost_of_singleton(x) for x in eles]

            cur_idx = 1
            for i in range(0, len(eles)):
                eles.insert(cur_idx, -1)
                cur_idx = cur_idx + 2

            cur_idx = 1
            for i in range(0, len(costs)):
                costs.insert(cur_idx, eps)
                cur_idx = cur_idx + 2

            cumsum_costs = list(accumulate(costs, initial=None))
            return cumsum_costs, eles

        csc_outside, ele_outside = outside_cumsum_costs(f)

        """
        sum = 0
        for i in range(0, min(100, len(ele_outside))):
            sum += f({ele_outside[i]})
            if ele_outside[i] >= 0:
                prev = 0.
                if i > 0:
                    prev = f(set(ele_outside[:i-1]))
                print(f"f(A):{f(set(ele_outside[:i+1]))},sum:{sum},sin:{f({ele_outside[i]})},sub:{f(set(ele_outside[:i+1])) - prev},ele:{ele_outside[i]}")
        """

        def G_over_N(x: float):
            idx = bisect.bisect_right(csc_outside, x)
            # print(f"in g, idx:{idx}, x:{x}")
            if idx == 0:
                return local_density(f, set(), ele_outside[0]) * x
            else:
                return f(set(ele_outside[:idx])) \
                       + local_density(f, set(ele_outside[:idx]), ele_outside[idx]) * (x - csc_outside[idx])

        def f_s(start: float):
            ks = bisect.bisect_right(csc_outside, start)
            if ks >= len(ele_outside):
                return 0

            slice_len = min(csc_outside[ks] - start, eps)
            val = local_density(f, set(), ele_outside[ks]) * slice_len

            return val

        delta = 0.

        start_from_new_ele = False

        ele_idx = 0
        prev_i_star = 0

        result = []

        budget_consumed = 0

        # print(f"new turn, csc:{csc_outside[:10]}")

        while True:
            v_j = 0.

            rbd = 0.

            terminate_signal = False

            # print(f"b:{b}, budget consumed:{budget_consumed}, eps:{eps}")
            if budget_consumed + eps <= b:
                # first, check the start point
                if start_from_new_ele:
                    ele_idx = ele_idx + 2
                    if ele_idx >= len(csc_outside) - 1:
                        # print("end 1")
                        break
                    stp_idx = csc_outside[ele_idx-1] + eps
                else:
                    stp_idx = prev_i_star + eps

                if ele_idx >= len(csc_outside) - 1:
                    # print("end 2")
                    break

                stp_left = f_s(stp_idx - eps)
                stp_right = G_over_N(stp_idx) - delta

                stp_value = min(stp_left, stp_right)
                stp_for_cur_ele = csc_outside[ele_idx]

                check_subsequent_elements = False

                rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)

                # print(f"start from left:{stp_left}, right:{stp_right}, stp_idx:{stp_idx}, g:{G_over_N(stp_idx)}, delta:{delta}")

                if stp_left <= stp_right and stp_idx <= stp_for_cur_ele:
                    v_j = stp_value
                    prev_i_star = stp_idx
                    start_from_new_ele = False
                    # print(f"stop at -1, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx}, c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
                elif stp_left <= stp_right and stp_idx > stp_for_cur_ele:
                    v_j = stp_value
                    prev_i_star = stp_idx
                    start_from_new_ele = True
                    check_subsequent_elements = True
                    # print(f"update at -1, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
                else:
                    # second, check the point where f sin = G N in this element
                    cur_marginal_density = local_density(f, set(ele_outside[:ele_idx]), ele_outside[ele_idx])
                    t_delta = (stp_left - stp_right) / cur_marginal_density
                    if stp_idx + t_delta <= stp_for_cur_ele:
                        # subsequent elements cannot bring higher value
                        v_j = stp_right + cur_marginal_density * t_delta
                        stp_idx = stp_idx + t_delta
                        prev_i_star = stp_idx
                        start_from_new_ele = False
                        rbd = eps
                        # print(f"update at 0, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
                        # print(f"stop at 0, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")
                    else:
                        stp_for_cur_ele_left = f_s(stp_for_cur_ele - eps)
                        stp_for_cur_ele_right = G_over_N(stp_for_cur_ele) - delta

                        cur_singleton_density = local_density(f, set(), ele_outside[ele_idx])
                        t_delta = (stp_for_cur_ele_left - stp_for_cur_ele_right)/cur_singleton_density
                        t_vj = stp_for_cur_ele_left - t_delta * cur_singleton_density

                        v_j = t_vj
                        stp_idx = stp_for_cur_ele + t_delta
                        prev_i_star = stp_idx
                        start_from_new_ele = True
                        rbd = csc_outside[ele_idx] - (stp_idx - eps)
                        # print(f"update at 1.5, ele {ele_idx}, left {stp_for_cur_ele_left}, right {stp_for_cur_ele_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]}) / 1}")

                        check_subsequent_elements = True

                if check_subsequent_elements:
                    # third, check the start point and the point where f sin = G N in subsequent elements until
                    # the element whose start point satisfying f sin <= G N
                    while True:
                        # start point
                        ele_idx = ele_idx + 2
                        if ele_idx >= len(csc_outside) - 1:
                            # print(f"stop at 1, ele {ele_idx}, left {stp_left}, right {stp_right}, v j {v_j}, idx:{stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]})/1}")
                            break

                        t_stp_idx = csc_outside[ele_idx-1] + eps
                        t_stp_left = f_s(t_stp_idx - eps)
                        t_stp_right = G_over_N(t_stp_idx) - delta

                        t_v_j = min(t_stp_left, t_stp_right)
                        if t_v_j >= v_j:
                            v_j = t_v_j
                            stp_idx = t_stp_idx
                            prev_i_star = stp_idx
                            rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                            if stp_idx + eps >= csc_outside[ele_idx]:
                                start_from_new_ele = True
                            else:
                                start_from_new_ele = False
                            #print(f"update at 2, ele {ele_idx}, left {t_stp_left}, right {t_stp_right}, v j {v_j}, idx:{t_stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]})/1}")

                        if t_stp_left <= t_stp_right:
                            #print(f"stop at 2, ele {ele_idx}, left {t_stp_left}, right {t_stp_right}, v j {v_j}, idx:{t_stp_idx},  c {csc_outside[ele_idx]}, density:{f({ele_outside[ele_idx]})/1}")
                            break

                        # balance point
                        cur_marginal_density = local_density(f, set(ele_outside[:ele_idx]), ele_outside[ele_idx])
                        stp_for_cur_ele = csc_outside[ele_idx]

                        t_delta = (t_stp_left - t_stp_right) / cur_marginal_density
                        if t_stp_idx + t_delta <= stp_for_cur_ele:
                            # subsequent elements cannot bring higher value
                            t_v_j = t_stp_right + cur_marginal_density * t_delta

                            v_j = t_v_j
                            stp_idx = t_stp_idx
                            prev_i_star = stp_idx
                            rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                            if stp_idx + eps >= csc_outside[ele_idx]:
                                start_from_new_ele = True
                            else:
                                start_from_new_ele = False
                            break
                        else:
                            stp_for_cur_ele_left = f_s(stp_for_cur_ele - eps)
                            stp_for_cur_ele_right = G_over_N(stp_for_cur_ele) - delta

                            cur_singleton_density = local_density(f, set(), ele_outside[ele_idx])
                            t_delta = (stp_for_cur_ele_left - stp_for_cur_ele_right) / cur_singleton_density
                            t_vj = stp_for_cur_ele_left - t_delta * cur_singleton_density

                            v_j = t_vj
                            stp_idx = stp_for_cur_ele + t_delta
                            rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                            prev_i_star = stp_idx
                            start_from_new_ele = True
            else:
                # start point
                # check the zero point of each element
                # if left > right , check balance point, update forward to next element
                # else update, terminate
                c_eps = b - budget_consumed
                if ele_idx >= len(csc_outside) - 1:
                    break
                # first, check the start point
                if start_from_new_ele:
                    ele_idx = ele_idx + 2
                    if ele_idx >= len(csc_outside) - 1:
                        break

                v_j = c_eps * f_s(csc_outside[ele_idx] - eps)/eps
                rbd = c_eps
                # print(f"end here? v_j:{v_j}, rbd:{rbd},ceps:{c_eps}")
                terminate_signal = True

                # stp_left = f_s(stp_idx - eps)
                # stp_right = G_over_N(stp_idx) - delta
                #
                # stp_value = min(stp_left, stp_right)
                # stp_for_cur_ele = csc_outside[ele_idx]
                #
                # rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                #
                # if stp_left <= stp_right:
                #     v_j = stp_value
                #     prev_i_star = stp_idx
                # else:
                #     # check the point where f sin = G N in this element
                #     stp_for_cur_ele_left = f_s(stp_for_cur_ele - eps)
                #     stp_for_cur_ele_right = G_over_N(stp_for_cur_ele) - delta
                #
                #     cur_singleton_density = local_density(f, set(), ele_outside[ele_idx])
                #     t_delta = (stp_for_cur_ele_left - stp_for_cur_ele_right)/cur_singleton_density
                #     t_vj = stp_for_cur_ele_left - t_delta * cur_singleton_density
                #     if t_vj >= v_j:
                #         v_j = t_vj
                #         stp_idx = stp_for_cur_ele + t_delta
                #         prev_i_star = stp_idx
                #         rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                #
                # while True:
                #     ele_idx = ele_idx + 2
                #     if ele_idx >= len(csc_outside) - 1:
                #         break
                #
                #     t_stp_idx = csc_outside[ele_idx] - c_eps + eps
                #     t_stp_left = f_s(t_stp_idx - eps)
                #     t_stp_right = G_over_N(t_stp_idx) - delta
                #
                #     t_v_j = min(t_stp_left, t_stp_right)
                #     if t_v_j >= v_j:
                #         v_j = t_v_j
                #         stp_idx = t_stp_idx
                #         prev_i_star = stp_idx
                #         rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)
                #
                #     if t_stp_left <= t_stp_right:
                #         break
                #
                #     # balance point
                #     stp_for_cur_ele_left = f_s(stp_for_cur_ele - eps)
                #     stp_for_cur_ele_right = G_over_N(stp_for_cur_ele) - delta
                #
                #     cur_singleton_density = local_density(f, set(), ele_outside[ele_idx])
                #     t_delta = (stp_for_cur_ele_left - stp_for_cur_ele_right) / cur_singleton_density
                #     t_vj = stp_for_cur_ele_left - t_delta * cur_singleton_density
                #     if t_vj >= v_j:
                #         v_j = t_vj
                #         stp_idx = stp_for_cur_ele + t_delta
                #         prev_i_star = stp_idx
                #         rbd = min(csc_outside[ele_idx], stp_idx) - (stp_idx - eps)

            delta += v_j
            budget_consumed += rbd

            # print(f"cur bd:{budget_consumed},rbd:{rbd},  vj:{v_j}, delta:{delta}, i star:{prev_i_star}, ele {ele_idx}")

            result.append((budget_consumed, delta))

            if terminate_signal:
                break

        #print(f"Method 3 completes with delta:{delta}")

        return result

    ub = 0

    M_plus_res = method3(f_over_base, model.budget)

    M_plus_budget = [x[0] for x in M_plus_res]
    M_plus_gain = [x[1] for x in M_plus_res]

    return max(M_plus_gain)

    def M_plus(x):
        idx = bisect.bisect_left(M_plus_budget, x) - 1
        if idx < 0:
            return x * M_plus_gain[0] / M_plus_budget[0]
        elif idx >= len(M_plus_gain)-1:
            return M_plus_gain[len(M_plus_gain)-1]
        else:
            return M_plus_gain[idx] + (x - M_plus_budget[idx]) * (M_plus_gain[idx + 1] - M_plus_gain[idx]) / (
                        M_plus_budget[idx + 1] - M_plus_budget[idx])


    endpoints_plus = [x[0] for x in M_plus_res]

    minimal_budget = model.budget - model.cost_of_set(base_set)
    cost_baseset = model.cost_of_set(base_set)

    endpoints = [0.]

    csc_inside, ele_inside = inside_cumsum_costs()
    endpoints_minus = csc_inside[:bisect.bisect_right(csc_inside, cost_baseset)]
    endpoints_minus = [x + minimal_budget for x in endpoints_minus]

    endpoints += list(set(endpoints_plus) | set(endpoints_minus))

    endpoints.append(model.budget)

    endpoints = list(set(endpoints))

    endpoints.sort()

    for i in range(0, len(endpoints)):
        if endpoints[i] >= minimal_budget:
            g_minus = G_minus(endpoints[i] - minimal_budget, model, set(model.ground_set), csc_inside, ele_inside)
            t_ub = M_plus(endpoints[i]) - g_minus
            # print(f"t ub is:{t_ub}, g minus is:{g_minus}, y is:{endpoints[i] - model.budget + base_budget},t:{model.budget-base_budget},b:{base_budget}")
            if t_ub > ub:
                ub = t_ub

    # print(f"delta is:{ub}, max budget:{max(M_plus_budget)}")

    return ub #max(M_plus_gain)

def marginal_delta_version6(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    c1 = model.budget

    base_set_value = model.objective(base_set)

    n = 1

    eps = c1 / n

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_density(x, base_set), reverse=False)
        costs = [model.cost_of_singleton(x) for x in s]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, s

    def local_density(f, base, ele):
        if ele == -1:
            return 0
        return (f(base | {ele}) - f(base)) / model.cost_of_singleton(ele)

    total_utility = model.objective(model.ground_set)

    def f_over_base(s):
        return model.objective(base_set | s) - base_set_value

    def f_minus(s):
        return total_utility - model.objective(set(model.ground_set) - set(s))

    def method3(f, b):
        def outside_cumsum_costs(f, b):
            t = list(remaining_set)
            t.append(-1)
            # sort density in descending order
            t.sort(key=lambda x: local_density(f, set(), x), reverse=True)

            costs = [model.cost_of_singleton(x) for x in t]
            costs.append(b)

            cumsum_costs = list(accumulate(costs, initial=None))
            return cumsum_costs, t

        csc_outside, ele_outside = outside_cumsum_costs(f, b)

        endpoints_enter = [0] + csc_outside

        endpoints_leave = [x + eps for x in endpoints_enter]

        endpoints = list(set(endpoints_leave + endpoints_enter))

        endpoints.sort()

        def G_over_N(x: float):
            idx = bisect.bisect_right(csc_outside, x) - 1
            if idx == -1:
                return local_density(f, set(), ele_outside[0]) * x
            else:
                return f(set(ele_outside[:idx])) \
                       + local_density(f, set(ele_outside[:idx]), ele_outside[idx]) * (x - csc_outside[idx])

        def f_s(start: float, eps: float):
            start_idx = bisect.bisect_right(csc_outside, start)
            stop_idx = bisect.bisect_left(csc_outside, start + eps)

            if start_idx >= len(ele_outside):
                return 0

            if start_idx == stop_idx:
                return local_density(f, set(), ele_outside[start_idx]) * eps
            else:
                slice_1_len = csc_outside[start_idx] - start
                val = local_density(f, set(), ele_outside[start_idx]) * slice_1_len
                eps -= slice_1_len

                cur_idx = start_idx + 1

                while eps > 0:
                    cur_ele_cost = model.cost_of_singleton(ele_outside[cur_idx])
                    if cur_ele_cost <= eps:
                        val += local_density(f, set(), ele_outside[cur_idx]) * cur_ele_cost

                        eps -= cur_ele_cost
                        cur_idx += 1

                    else:
                        val += local_density(f, set(), ele_outside[cur_idx]) * eps
                        break

                return val

        delta = 0.
        endpoint_idx = 0
        prev_i_star = 0

        result = []

        budget_consumed = 0

        while budget_consumed < b:
            c_eps = min(eps, b - budget_consumed)

            v_j = 0.

            while True:
                ele_idx = bisect.bisect_left(csc_outside, endpoints[endpoint_idx])

                if endpoints[endpoint_idx] - prev_i_star >= c_eps:
                    left = G_over_N(endpoints[endpoint_idx]) - delta

                    right = f_s(endpoints[endpoint_idx] - c_eps, c_eps)

                    if left - right >= 0:
                        if ele_idx == 0:
                            left_k = local_density(f, set(), ele_outside[ele_idx])
                        else:
                            left_k = local_density(f, set(ele_outside[:ele_idx - 1]), ele_outside[ele_idx])

                        if endpoint_idx == 0:
                            right_k = right / endpoints[0]
                        else:
                            prev_right = f_s(endpoints[endpoint_idx - 1] - c_eps, c_eps)
                            right_k = (right - prev_right) / (csc_outside[endpoint_idx] - csc_outside[endpoint_idx - 1])

                        i_star = max(endpoints[endpoint_idx] - (left - right) / (left_k - right_k),
                                     prev_i_star + c_eps)

                        v_j = f_s(i_star - c_eps, c_eps)

                        prev_i_star = i_star

                        break
                    else:
                        endpoint_idx += 1
                else:
                    endpoint_idx += 1

            delta += v_j
            budget_consumed += c_eps

            result.append((budget_consumed, delta))

        return result

    def method4(f, b):
        def outside_cumsum_costs(f, b):
            t = list(remaining_set)
            t.append(-1)
            # sort density in descending order
            t.sort(key=lambda x: local_density(f, set(), x), reverse=False)

            costs = [model.cost_of_singleton(x) for x in t]
            costs.append(b)

            cumsum_costs = list(accumulate(costs, initial=None))
            return cumsum_costs, t

        csc_outside, ele_outside = outside_cumsum_costs(f, b)

        endpoints_enter = [0] + csc_outside

        endpoints_leave = [x + eps for x in endpoints_enter]

        endpoints = list(set(endpoints_leave + endpoints_enter))

        endpoints.sort()

        def G_over_N(x: float):
            idx = bisect.bisect_right(csc_outside, x) - 1
            if idx == -1:
                return local_density(f, {}, ele_outside[0]) * x
            else:
                return f(set(ele_outside[:idx])) \
                       + local_density(f, set(ele_outside[:idx]), ele_outside[idx]) * (x - csc_outside[idx])

        def f_s(start: float, eps: float):
            start_idx = bisect.bisect_right(csc_outside, start)
            stop_idx = bisect.bisect_left(csc_outside, start + eps)

            if start_idx >= len(ele_outside):
                return 0

            if start_idx == stop_idx:
                return local_density(f, set(), ele_outside[start_idx]) * eps
            else:
                slice_1_len = csc_outside[start_idx] - start
                val = local_density(f, set(), ele_outside[start_idx]) * slice_1_len
                eps -= slice_1_len

                cur_idx = start_idx + 1

                while eps > 0:
                    cur_ele_cost = model.cost_of_singleton(ele_outside[cur_idx])
                    if cur_ele_cost <= eps:
                        val += local_density(f, set(), ele_outside[cur_idx]) * cur_ele_cost

                        eps -= cur_ele_cost
                        cur_idx += 1

                    else:
                        val += local_density(f, set(), ele_outside[cur_idx]) * eps
                        break

                return val

        delta = 0.
        endpoint_idx = 0
        prev_i_star = 0

        result = []

        budget_consumed = 0

        while budget_consumed < b:
            c_eps = min(eps, b - budget_consumed)

            v_j = 0.

            while True:
                ele_idx = bisect.bisect_left(csc_outside, endpoints[endpoint_idx])

                if endpoints[endpoint_idx] - prev_i_star >= c_eps:
                    left = G_over_N(endpoints[endpoint_idx]) - delta

                    right = f_s(endpoints[endpoint_idx] - c_eps, c_eps)

                    if left - right >= 0:
                        if ele_idx == 0:
                            left_k = local_density(f, {}, ele_outside[ele_idx])
                        else:
                            left_k = local_density(f, set(ele_outside[:ele_idx - 1]), ele_outside[ele_idx])

                        if endpoint_idx == 0:
                            right_k = right / endpoints[0]
                        else:
                            prev_right = f_s(endpoints[endpoint_idx - 1] - c_eps, c_eps)
                            right_k = (right - prev_right) / (csc_outside[endpoint_idx] - csc_outside[endpoint_idx - 1])

                        i_star = max(endpoints[endpoint_idx] - (left - right) / (left_k - right_k),
                                     prev_i_star + c_eps)

                        v_j = f_s(i_star - c_eps, c_eps)

                        prev_i_star = i_star

                        break
                    else:
                        endpoint_idx += 1
                else:
                    endpoint_idx += 1

            delta += v_j
            budget_consumed += c_eps

            result.append((budget_consumed, delta))

        return result

    ub = 0

    M_plus_res = method3(f_over_base, model.budget)

    M_plus_budget = [x[0] for x in M_plus_res]
    M_plus_gain = [x[1] for x in M_plus_res]

    def M_plus(x):
        idx = bisect.bisect_left(M_plus_budget, x) - 1
        if idx < 0:
            return x * M_plus_gain[0] / M_plus_budget[0]
        else:
            return M_plus_gain[idx] + (x - M_plus_budget[idx]) * (M_plus_gain[idx + 1] - M_plus_gain[idx]) / (
                        M_plus_budget[idx + 1] - M_plus_budget[idx])


    endpoints_plus = [x[0] for x in M_plus_res]

    minimal_budget = model.budget - model.cost_of_set(base_set)
    cost_baseset = model.cost_of_set(base_set)

    endpoints = [0.]

    csc_inside, ele_inside = inside_cumsum_costs()
    endpoints_minus = csc_inside[:bisect.bisect_right(csc_inside, cost_baseset)]
    endpoints_minus = [x + minimal_budget for x in endpoints_minus]

    endpoints += list(set(endpoints_plus) | set(endpoints_minus))

    endpoints.append(model.budget)

    endpoints = list(set(endpoints))

    endpoints.sort()

    for i in range(0, len(endpoints)):
        if endpoints[i] >= minimal_budget:
            g_minus = G_minus(endpoints[i] - minimal_budget, model, set(model.ground_set), csc_inside, ele_inside)
            t_ub = M_plus(endpoints[i]) - g_minus
            #print(f"t ub is:{t_ub}, g minus is:{g_minus}, y is:{endpoints[i] - model.budget + base_budget},t:{model.budget-base_budget},b:{base_budget}")
            if t_ub > ub:
                ub = t_ub

    #print(f"delta is:{ub}, max is:{M_plus(model.budget)}")

    return ub

def marginal_delta_version7(base_set: Set[int], remaining_set: Set[int], model: BaseTask, minus = False):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    if len(remaining_set) == 0:
        return 0

    base_set_value = model.objective(base_set)

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_density(x, base_set), reverse=False)
        costs = [model.cost_of_singleton(x) for x in s]
        cumsum_costs = list(accumulate(costs, initial=None))
        return cumsum_costs, s

    def local_f(S):
        S = list(S)
        while S.count(-1) > 0:
            S.remove(-1)
        return model.objective(S)

    def local_density(f, base, ele):
        return (f(base | {ele}) - f(base)) / model.cost_of_singleton(ele)

    total_utility = model.objective(model.ground_set)

    def f_over_base(s):
        return local_f(base_set | s) - base_set_value

    def method3(f, b):
        def outside_cumsum_costs(f):
            eles = list(remaining_set)
            # sort density in descending order
            eles.sort(key=lambda x: local_density(f, set(), x), reverse=True)
            costs = [model.cost_of_singleton(x) for x in eles]

            cumsum_costs = list(accumulate(costs, initial=None))
            return cumsum_costs, eles

        csc_outside, ele_outside = outside_cumsum_costs(f)

        delta = 0.

        ele_idx = 0

        result = []

        budget_consumed = 0

        while budget_consumed < b and ele_idx < len(ele_outside):
            marginal_gain = f(set(ele_outside[:ele_idx+1])) - f(set(ele_outside[:ele_idx]))
            singleton_gain = f({ele_outside[ele_idx]})

            if singleton_gain <= 0:
                budget_consumed = b
                result.append((budget_consumed, delta))
                break

            budget_to_use_up = (marginal_gain / singleton_gain) * model.cost_of_singleton(ele_outside[ele_idx])

            if b - budget_consumed >= budget_to_use_up:
                v_j = marginal_gain
                bd = budget_to_use_up
            else:
                v_j = (b-budget_consumed) * (singleton_gain/model.cost_of_singleton(ele_outside[ele_idx]))
                bd = b - budget_consumed

            ele_idx = ele_idx + 1

            delta += v_j
            budget_consumed += bd

            result.append((budget_consumed, delta))

        return result

    ub = 0

    M_plus_res = method3(f_over_base, model.budget)

    M_plus_budget = [x[0] for x in M_plus_res]
    M_plus_gain = [x[1] for x in M_plus_res]

    if not minus:
        return max(M_plus_gain)

    def M_plus(x):
        idx = bisect.bisect_left(M_plus_budget, x) - 1
        if idx < 0:
            return x * M_plus_gain[0] / M_plus_budget[0]
        elif idx >= len(M_plus_gain)-1:
            return M_plus_gain[len(M_plus_gain)-1]
        else:
            return M_plus_gain[idx] + (x - M_plus_budget[idx]) * (M_plus_gain[idx + 1] - M_plus_gain[idx]) / (
                        M_plus_budget[idx + 1] - M_plus_budget[idx])


    endpoints_plus = [x[0] for x in M_plus_res]

    minimal_budget = model.budget - model.cost_of_set(base_set)
    cost_baseset = model.cost_of_set(base_set)

    endpoints = [minimal_budget]

    csc_inside, ele_inside = inside_cumsum_costs()

    endpoints_minus = csc_inside[:bisect.bisect_right(csc_inside, cost_baseset)]
    endpoints_minus = [x + minimal_budget for x in endpoints_minus]

    endpoints += list(set(endpoints_plus) | set(endpoints_minus))

    endpoints.append(model.budget)

    endpoints.sort()

    # print(f"csc:{csc_inside}, S:{base_set}, 1:{model.objective(model.ground_set)}, 2:{model.objective(list(set(model.ground_set) - {41,63}))}, 3:{model.objective(model.ground_set) - model.objective(list(set(model.ground_set) - {41,63}))}, 4:{model.objective({41,63})}")

    for i in range(0, len(endpoints)):
        if endpoints[i] >= minimal_budget:
            g_plus = M_plus(endpoints[i])
            g_minus = G_minus(endpoints[i] - minimal_budget, model, set(model.ground_set), csc_inside, ele_inside)
            t_ub = g_plus - g_minus
            # print(f"t ub is:{t_ub}, g plus is:{g_plus}, g minus is:{g_minus}, x is:{endpoints[i]}, y is:{endpoints[i] - minimal_budget},t:{model.budget-cost_baseset},b:{cost_baseset},f(S):{model.objective(base_set)}")
            if t_ub > ub:
                ub = t_ub

    # print(f"delta is:{ub}, max budget:{max(M_plus_budget)}")

    return ub

def marginal_delta_for_streaming_version1(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    """Delta( b | S )"""
    assert len(base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    k = model.budget

    window = []

    for i in remaining_set:
        accepted = False
        for j in range(0, len(window)):
            if model.objective([i]) >= model.objective([window[j]]) and model.cost_of_singleton(i) <= model.budget:
                window.insert(j, i)
                accepted = True
                break

        if len(window) < k:
            if not accepted:
                window.append(i)
        else:
            if accepted:
                window.pop()

    return model.objective(base_set) + sum([model.objective([i]) for i in window])


def marginal_delta_for_knapsack_streaming_version1(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    """Delta( b | S )"""
    assert len(base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    b = model.budget

    window = []
    total_cost = 0

    for i in remaining_set:
        accepted = False
        for j in range(0, len(window)):
            if model.density(i, list(base_set)) >= model.density(window[j][0], list(base_set)):
                window.insert(j, (i, model.cost_of_singleton(i)))
                total_cost += model.cost_of_singleton(i)
                accepted = True
                break

        if total_cost < b:
            if not accepted:
                window.append((i, model.cost_of_singleton(i)))
                total_cost += model.cost_of_singleton(i)
        else:
            if accepted:
                while total_cost > b:
                    out = window.pop()
                    if total_cost - out[1] <= b:
                        cut_value = total_cost - b
                        total_cost = b
                        window.append((out[0], out[1] - cut_value))
                        break
                    else:
                        total_cost -= out[1]

    return model.objective(base_set) + sum([model.density(i[0], base_set) * i[1] for i in window])

# c means cardinality
def marginal_delta_version4_c(base_set: Set[int], remaining_set: Set[int], model: BaseTask, minus = False):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    base_set_value = model.objective(base_set)

    step = 1

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_marginal_gain(x), reverse=False)

        values = [model.cutout_marginal_gain(x) for x in s]
        g_m = list(accumulate(values, initial=None))
        return g_m, s


    def local_objective(f, base, ele):
        return f(base | {ele}) - f(base)

    def f_over_base(s):
        return model.objective(base_set | s) - base_set_value

    def outside_cumsum_costs(f, b):
        t = list(remaining_set)
        # sort density in descending order
        t.sort(key=lambda x: local_objective(f, set(), x), reverse=True)
        for i in range(0, step):
            t.append(-1)

        values = [local_objective(f, set(), x) for x in t]
        g_p = list(accumulate(values))
        return g_p, t

    g_p, ele_outside = outside_cumsum_costs(f_over_base, model.budget)

    def method3(f, b):
        def G_over_N(x: int):
            x = int(x)
            return f(set(ele_outside[:x+1]))

        def f_s(stop: int,  step: int):
            if stop >= len(ele_outside):
                return 0

            stop = int(stop)
            step = int(step)

            cur_idx = stop
            start = cur_idx - step
            val = 0

            while cur_idx > start and cur_idx >= 0:
                val += local_objective(f, set(), ele_outside[cur_idx])
                cur_idx -= 1

            return val

        delta = 0.
        ele_idx = 0
        prev_i_star = -1

        result = []

        budget_consumed = 0

        while budget_consumed < b:
            while True:
                if ele_idx - prev_i_star >= 1:
                    left = G_over_N(ele_idx) - delta
                    right = f_s(ele_idx, 1)
                    #print(f"ex left:{left},right:{right},ele:{ele_idx},prev:{prev_i_star},c:{c_step}")
                    if left - right >= 0:
                        i_star = 0
                        if ele_idx - 1 - prev_i_star < 1:
                            i_star = ele_idx
                            #print(f"updated 0:{v_j}, left:{left},right:{right},ele:{ele_idx}")
                            v_j = right
                        else:
                            prev_left = G_over_N(ele_idx-1) - delta
                            if prev_left > right:
                                i_star = ele_idx - 1
                                v_j = prev_left
                            else:
                                i_star = ele_idx
                                v_j = right

                        prev_i_star = i_star
                        break
                    else:
                        ele_idx += 1
                else:
                    ele_idx = prev_i_star + 1

            delta += v_j
            budget_consumed += 1

            # print(f"budget_consumed:{budget_consumed}, delta:{delta}, vj:{v_j}, i star:{i_star}")

            result.append((budget_consumed, delta, i_star))

        return result

    M_plus_res = method3(f_over_base, model.budget)

    M_plus_budget = [x[0] for x in M_plus_res]
    M_plus_gain = [x[1] for x in M_plus_res]
    M_plus_ele_idx = [x[2] for x in M_plus_res]

    if minus == False:
        return max(M_plus_gain)

    ub = 0.

    def M_plus(x):
        if x <= 0:
            return 0
        x = int(x)
        idx = bisect.bisect_left(M_plus_budget, x) - 1
        if idx >= len(M_plus_gain):
            return M_plus_gain[len(M_plus_gain) - 1]

        offset = x - M_plus_budget[idx]
        if idx < 0:
            return min(g_p[x-1], M_plus_gain[0])
        else:
            return min(M_plus_gain[idx] + g_p[min(offset + M_plus_ele_idx[idx],len(g_p)-1)] - g_p[min(M_plus_ele_idx[idx], len(g_p)-1)], M_plus_gain[idx + 1])

    endpoints_plus = [x[0] for x in M_plus_res]

    cost_baseset = model.cost_of_set(base_set)
    minimal_budget = model.budget - cost_baseset

    endpoints = [minimal_budget]

    g_m, ele_inside = inside_cumsum_costs()

    endpoints_minus = [x + minimal_budget for x in range(0, int(cost_baseset))]

    endpoints += list(set(endpoints_plus) | set(endpoints_minus))

    endpoints.append(model.budget)

    endpoints = list(set(endpoints))

    endpoints.sort()

    #print(f"g_p:{g_p[:10]}, m b:{M_plus_budget}, m g:{M_plus_gain}, g_m:{g_m}")

    for i in range(0, len(endpoints)):
        if endpoints[i] >= minimal_budget:
            g_minus = 0
            if endpoints[i] > minimal_budget:
                g_minus = g_m[int(endpoints[i] - minimal_budget)-1]
            t_ub = M_plus(endpoints[i]) - g_minus
            #print(f"t ub:{t_ub}, g p:{M_plus(endpoints[i])}, g m:{g_minus},idx:{int(endpoints[i] - minimal_budget) - 1},eid:{endpoints[i]}")
            if t_ub > ub:
                #print(f"t ub:{t_ub}, g p:{M_plus(endpoints[i])}, g m:{g_minus},idx:{int(endpoints[i] - minimal_budget)-1},eid:{endpoints[i]}")
                ub = t_ub

    return ub

def marginal_delta_version5_c(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    base_set_value = model.objective(base_set)

    step = 2

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_marginal_gain(x, model.ground_set), reverse=False)

        values = [model.cutout_marginal_gain(x, model.ground_set) for x in s]
        g_m = list(accumulate(values, initial=None))
        return g_m, s

    def local_objective(f, base, ele):
        return f(base | {ele}) - f(base)

    def f_over_base(s):
        return model.objective(base_set | s) - base_set_value

    def outside_cumsum_costs(f, b):
        t = list(remaining_set)
        # sort pair value in descending order
        t.sort(key=lambda x: model.max_2_pair[x], reverse=True)
        for i in range(0, step):
            t.append(-1)

        t_sin = list(remaining_set)
        t_sin.sort(key=lambda x: local_objective(f, set(), x), reverse=True)

        values = [local_objective(f, set(), x) for x in t_sin]

        g_p = list(accumulate(values))
        return g_p, t

    g_p, ele_outside = outside_cumsum_costs(f_over_base, model.budget)

    def method3(f, b):
        def G_over_N(x: int):
            x = int(x)
            return f(set(ele_outside[:x+1]))

        def I_N(stop: int):
            if stop >= len(ele_outside):
                return 0

            stop = int(stop)
            return model.max_2_pair[ele_outside[stop]]

        delta = 0.
        ele_idx = 0
        prev_i_star = -1

        result = []

        budget_consumed = 0

        while budget_consumed < b:
            c_step = min(step, b - budget_consumed)

            v_j = 0.

            while True:
                if ele_idx - prev_i_star >= c_step:
                    left = G_over_N(ele_idx) - delta
                    right = I_N(ele_idx)
                    #print(f"ex left:{left},right:{right},ele:{ele_idx},prev:{prev_i_star},c:{c_step}")
                    if left - right >= 0:
                        i_star = 0
                        if ele_idx - 1 - prev_i_star < c_step:
                            i_star = ele_idx
                            #print(f"updated 0:{v_j}, left:{left},right:{right},ele:{ele_idx}")
                            v_j = right
                        else:
                            prev_left = G_over_N(ele_idx-1) - delta
                            if prev_left > right:
                                i_star = ele_idx - 1
                                #print(f"updated 1:{v_j}, left:{left},right:{right},ele:{ele_idx}, prev_left:{prev_left}")
                                v_j = prev_left
                            else:
                                #print(f"updated 2:{v_j}, left:{left},right:{right},ele:{ele_idx}, prev_left:{prev_left}")
                                i_star = ele_idx
                                v_j = right

                        prev_i_star = i_star
                        break
                    else:
                        ele_idx += 1
                else:
                    ele_idx = prev_i_star + c_step

            delta += v_j
            budget_consumed += c_step

            #print(f"budget_consumed:{budget_consumed}, delta:{delta}, vj:{v_j}, i star:{i_star}")

            result.append((budget_consumed, delta, i_star))

        return result

    M_plus_res = method3(f_over_base, model.budget)

    M_plus_budget = [x[0] for x in M_plus_res]
    M_plus_gain = [x[1] for x in M_plus_res]
    M_plus_ele_idx = [x[2] for x in M_plus_res]

    ub = 0.

    def M_plus(x):
        if x <= 0:
            return 0
        x = int(x)
        idx = bisect.bisect_left(M_plus_budget, x) - 1
        if idx >= len(M_plus_gain):
            return M_plus_gain[len(M_plus_gain) - 1]

        offset = x - M_plus_budget[idx]
        if idx < 0:
            return min(g_p[x-1], M_plus_gain[0])
        else:
            return min(M_plus_gain[idx] + g_p[min(offset + M_plus_ele_idx[idx], len(g_p)-1)] - g_p[min(M_plus_ele_idx[idx], len(g_p)-1)], M_plus_gain[idx + 1])

    endpoints_plus = [x[0] for x in M_plus_res]

    cost_baseset = model.cost_of_set(base_set)

    minimal_budget = model.budget - cost_baseset

    endpoints = [0.]

    g_m, ele_inside = inside_cumsum_costs()

    endpoints_minus = [x + minimal_budget for x in range(0, cost_baseset)]

    endpoints += list(set(endpoints_plus) | set(endpoints_minus))

    endpoints.append(model.budget)

    endpoints = list(set(endpoints))

    endpoints.sort()

    for i in range(0, len(endpoints)):
        if endpoints[i] >= minimal_budget:
            g_minus = 0
            if endpoints[i] > minimal_budget:
                g_minus = g_m[int(endpoints[i] - minimal_budget)-1]
            t_ub = M_plus(endpoints[i]) - g_minus
            if t_ub > ub:
                ub = t_ub

    return ub

def marginal_delta_version6_c(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    assert len(
        base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)

    base_set_value = model.objective(base_set)

    step = 2

    def inside_cumsum_costs():
        s = list(base_set)
        # sort density in ascending order, default sort has ascending order
        s.sort(key=lambda x: model.cutout_marginal_gain(x, model.ground_set), reverse=False)

        values = [model.cutout_marginal_gain(x, model.ground_set) for x in s]
        g_m = list(accumulate(values, initial=None))
        return g_m, s

    def local_objective(f, base, ele):
        return f(base | {ele}) - f(base)

    def f_over_base(s):
        return model.objective(base_set | s) - base_set_value

    def outside_cumsum_costs(f, b):
        t = list(remaining_set)
        # sort density in descending order
        t.sort(key=lambda x: local_objective(f, set(), x), reverse=True)
        for i in range(0, step):
            t.append(-1)

        values = [local_objective(f, set(), x) for x in t]
        g_p = list(accumulate(values))
        return g_p, t

    g_p, ele_outside = outside_cumsum_costs(f_over_base, model.budget)

    def method3(f, b):
        def G_over_N(x: int):
            x = int(x)
            return f(set(ele_outside[:x+1]))

        def f_s(stop: int,  prev: int):
            if stop >= len(ele_outside):
                return 0
            stop = int(stop)
            prev = int(prev)
            return f({ele_outside[stop]}) + f({ele_outside[prev]})

        delta = 0.
        ele_idx = 0
        prev_i_star = -1

        result = []

        budget_consumed = 0

        ele_idx_used = 0

        while budget_consumed < b:
            c_step = min(step, b - budget_consumed)

            v_j = 0.

            while True:
                if ele_idx - prev_i_star >= c_step:
                    left = G_over_N(ele_idx) - delta
                    right = f_s(ele_idx, prev_i_star + 1)
                    #print(f"ex left:{left},right:{right},ele:{ele_idx},prev:{prev_i_star},c:{c_step}")
                    if left - right >= 0:
                        i_star = 0
                        if ele_idx - 1 - prev_i_star < c_step:
                            i_star = ele_idx
                            #print(f"updated 0:{v_j}, left:{left},right:{right},ele:{ele_idx}")
                            v_j = right
                        else:
                            prev_left = G_over_N(ele_idx-1) - delta
                            if prev_left > right:
                                i_star = ele_idx - 1
                                #print(f"updated 1:{v_j}, left:{left},right:{right},ele:{ele_idx}, prev_left:{prev_left}")
                                v_j = prev_left
                            else:
                                #print(f"updated 2:{v_j}, left:{left},right:{right},ele:{ele_idx}, prev_left:{prev_left}")
                                i_star = ele_idx
                                v_j = right

                        prev_i_star = i_star
                        break
                    else:
                        ele_idx += 1
                else:
                    ele_idx = prev_i_star + c_step

            delta += v_j
            budget_consumed += c_step

            print(f"budget_consumed:{budget_consumed}, delta:{delta}, vj:{v_j}, i star:{i_star}")

            result.append((budget_consumed, delta, i_star))

        return result

    M_plus_res = method3(f_over_base, model.budget)

    M_plus_budget = [x[0] for x in M_plus_res]
    M_plus_gain = [x[1] for x in M_plus_res]
    M_plus_ele_idx = [x[2] for x in M_plus_res]

    ub = 0.

    def M_plus(x):
        if x <= 0:
            return 0
        x = int(x)
        idx = bisect.bisect_left(M_plus_budget, x) - 1
        if idx >= len(M_plus_gain):
            return M_plus_gain[len(M_plus_gain) - 1]

        offset = x - M_plus_budget[idx]
        if idx < 0:
            return min(g_p[x-1], M_plus_gain[0])
        else:
            return min(M_plus_gain[idx] + g_p[min(offset + M_plus_ele_idx[idx],len(g_p)-1)] - g_p[min(M_plus_ele_idx[idx], len(g_p)-1)], M_plus_gain[idx + 1])

    endpoints_plus = [x[0] for x in M_plus_res]

    cost_baseset = model.cost_of_set(base_set)

    minimal_budget = model.budget - cost_baseset

    endpoints = [0.]

    g_m, ele_inside = inside_cumsum_costs()

    endpoints_minus = [x + minimal_budget for x in range(0, cost_baseset)]

    endpoints += list(set(endpoints_plus) | set(endpoints_minus))

    endpoints.append(model.budget)

    endpoints = list(set(endpoints))

    endpoints.sort()

    for i in range(0, len(endpoints)):
        if endpoints[i] >= minimal_budget:
            g_minus = 0
            if endpoints[i] > minimal_budget:
                g_minus = g_m[int(endpoints[i] - minimal_budget)-1]
            t_ub = M_plus(endpoints[i]) - g_minus
            if t_ub > ub:
                ub = t_ub

    return ub


def marginal_delta_version7_c(base_set: Set[int], remaining_set: Set[int], model: BaseTask):
    """Delta( b | S )"""
    assert len(base_set & remaining_set) == 0, "{} ----- {}".format(base_set, remaining_set)
    if len(remaining_set) == 0:
        return 0
    return 0

def marginal_delta_gate(upb: str, base_set, remaining_set, model:BaseTask):

    remaining_set = set(model.ground_set) - set(base_set)
    if upb is not None:
        delta = 0.
        if upb == "ub1":
            delta = marginal_delta(base_set, remaining_set, model)
        elif upb == "ub2":
            delta = marginal_delta_version2(base_set, remaining_set, model)
        elif upb == "ub3":
            delta = marginal_delta_version3(base_set, remaining_set, model)
        elif upb == 'ub4':
            delta = marginal_delta_version4(base_set, remaining_set, model)
        elif upb == 'ub4c':
            delta = marginal_delta_version4_c(base_set, remaining_set, model)
        elif upb == 'ub4cm':
            delta = marginal_delta_version4_c(base_set, remaining_set, model, minus=True)
        elif upb == 'ub5':
            delta = marginal_delta_version5(base_set, remaining_set, model)
        elif upb == 'ub5c':
            delta = marginal_delta_version5_c(base_set, remaining_set, model)
        elif upb == 'ub5p':
            delta = marginal_delta_version5_p(base_set, remaining_set, model)
        elif upb == 'ub6':
            delta = marginal_delta_version6_c(base_set, remaining_set, model)
        elif upb == 'ub7c':
            delta = marginal_delta_version7_c(base_set, remaining_set, model)
        elif upb == 'ub7':
            delta = marginal_delta_version7(base_set, remaining_set, model)
        elif upb == 'ub7m':
            delta = marginal_delta_version7(base_set, remaining_set, model, minus=True)

        else:
            raise ValueError("Unsupported Upperbound")
        return delta
    else:
        raise ValueError("Upperbound unassigned")