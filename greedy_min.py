"""
# algorithm
# implement upper bound mentioned in revisiting original paper
"""
import copy
import time

import optimizer
from base_task import BaseTask
from data_dependent_upperbound import marginal_delta_min_gate
from optimizer import UnifiedMinOptimizer


def simple_greedy_min(model: BaseTask, upb=None):
    start_time = time.time()
    res = {}

    G = set()
    remaining_elements = set(model.ground_set)
    lambda_capital = 0.
    if upb is not None:
        delta, parameters = marginal_delta_min_gate(upb, set({}), remaining_elements, model)
        lambda_capital = delta

    # print(f"model.value:{model.value}")
    while model.objective(list(G)) < model.value and len(remaining_elements) > 0:
        s, max_md = None, -1
        for e in remaining_elements:
            md = model.marginal_gain(e, list(G)) / model.cost_of_singleton(e)
            if s is None or md > max_md:
                s, max_md = e, md

        temp_G = G | {s}
        if model.objective(list(temp_G)) < model.value:
            G.add(s)
        else:
            min_cost = model.cost_of_singleton(s)
            min_s = s
            for e in remaining_elements:
                temp_G = G | {e}
                if model.objective(list(temp_G)) >= model.value:
                    c_e = model.cost_of_singleton(e)
                    if c_e < min_cost:
                        min_s, min_cost = e, c_e
            G.add(min_s)

        if upb is not None:
            delta, parameters = marginal_delta_min_gate(upb, G, remaining_elements, model)
            if lambda_capital < delta:
                lambda_capital = delta

        remaining_elements.remove(s)

    res['S'] = G
    res['f(S)'] = model.objective(list(G))
    res['c(S)'] = model.cost_of_set(list(G))
    res['target'] = model.value
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['c(S)'] / lambda_capital

    stop_time = time.time()
    res['Time'] = stop_time - start_time
    return res


def simple_greedy_min_opt(model: BaseTask, upb=None):
    start_time = time.time()
    res = {}

    G = set()
    remaining_elements = set(model.ground_set)
    opt = optimizer.MultilinearDualOptimizer()
    opt.setModel(model)

    opt.setBase([])
    opt.build()
    lambda_capital = opt.optimize()['lwb']

    while model.objective(list(G)) < model.value and len(remaining_elements) > 0:
        s, max_md = None, -1
        for e in remaining_elements:
            md = model.marginal_gain(e, list(G)) / model.cost_of_singleton(e)
            if s is None or md > max_md:
                s, max_md = e, md
        temp_G = G | {s}
        if model.objective(list(temp_G)) < model.value:
            G.add(s)
        else:
            min_cost = model.cost_of_singleton(s)
            min_s = s
            for e in remaining_elements:
                temp_G = G | {e}
                if model.objective(list(temp_G)) >= model.value:
                    c_e = model.cost_of_singleton(e)
                    if c_e < min_cost:
                        min_s, min_cost = e, c_e
            G.add(min_s)

        opt.setBase(G)
        opt.build()
        temp_lwb = opt.optimize()['lwb']
        if temp_lwb > lambda_capital:
            lambda_capital = temp_lwb

        remaining_elements.remove(s)

    res['S'] = G
    res['f(S)'] = model.objective(list(G))
    res['c(S)'] = model.cost_of_set(list(G))
    res['target'] = model.value
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['c(S)'] / lambda_capital

    stop_time = time.time()
    res['Time'] = stop_time - start_time
    return res


def augmented_greedy_min(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    res = {}

    G = set()
    remaining_elements = set(model.ground_set)
    lambda_capital = 0.
    if upb is not None:
        delta, parameters = marginal_delta_min_gate(upb, set({}), remaining_elements, model)
        lambda_capital = delta
        # print(f"l updated:{lambda_capital}, d:{delta} 1")

    while model.objective(list(G)) < model.value and len(remaining_elements) > 0:
        candidates = set()
        for e in remaining_elements:
            if model.objective(list(G | {e})) < model.value:
                candidates.add(e)

        if len(candidates) == 0:
            min_cost = -1
            min_s = None
            for e in remaining_elements:
                c_e = model.cost_of_singleton(e)
                if min_s is None or c_e < min_cost:
                    min_s, min_cost = e, c_e
            G.add(min_s)
        else:
            s, max_md = None, -1
            for e in candidates:
                md = model.marginal_gain(e, list(G)) / model.cost_of_singleton(e)
                if s is None or md > max_md:
                    s, max_md = e, md
            # print(f"1: s:{s}, md:{max_md}, md:{model.marginal_gain(419, list(G)) / model.cost_of_singleton(419)}")
            G.add(s)
            remaining_elements.remove(s)

        if upb is not None:
            delta, parameters = marginal_delta_min_gate(upb, G, remaining_elements, model)
            if lambda_capital < delta:
                lambda_capital = delta

    res['S'] = G
    res['f(S)'] = model.objective(list(G))
    res['c(S)'] = model.cost_of_set(list(G))
    res['target'] = model.value
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['c(S)'] / lambda_capital
        res['p'] = parameters

    stop_time = time.time()
    res['Time'] = stop_time - start_time
    return res


def simple_greedy_new_min(model: BaseTask, upb=None):
    start_time = time.time()
    res = {}

    G = set()
    remaining_elements = set(model.ground_set)
    lambda_capital = 0.

    opt = optimizer.DualOptimizer()
    opt.setModel(model)
    opt.addIntermediate(set())

    while model.objective(list(G)) < model.value and len(remaining_elements) > 0:
        s, max_md = None, -1
        for e in remaining_elements:
            md = model.marginal_gain(e, list(G)) / model.cost_of_singleton(e)
            if s is None or md > max_md:
                s, max_md = e, md
        temp_G = G | {s}
        if model.objective(list(temp_G)) < model.value:
            G.add(s)
        else:
            min_cost = model.cost_of_singleton(s)
            min_s = s
            for e in remaining_elements:
                temp_G = G | {e}
                if model.objective(list(temp_G)) >= model.value:
                    c_e = model.cost_of_singleton(e)
                    if c_e < min_cost:
                        min_s, min_cost = e, c_e
            G.add(min_s)

        if model.objective(list(G)) < model.value:
            # print(f"G:{model.objective(list(G))}")
            opt.addIntermediate(G)
        remaining_elements.remove(s)

    opt.build()
    lambda_capital = opt.optimize()['lwb']

    res['S'] = G
    res['f(S)'] = model.objective(list(G))
    res['c(S)'] = model.cost_of_set(list(G))
    res['target'] = model.value
    if upb is not None:
        res['Lambda'] = lambda_capital
        res['AF'] = res['c(S)'] / lambda_capital

    stop_time = time.time()
    res['Time'] = stop_time - start_time
    return res


def greedy_mintss(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    if upb is not None:
        delta, parameters = marginal_delta_min_gate(upb, set({}), remaining_elements, model)
        lambda_capital = delta
    # print("gonna go1")

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        if upb is not None:
            delta, parameters = marginal_delta_min_gate(upb, s, remaining_elements, model)
            if lambda_capital < delta:
                lambda_capital = delta

    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret


def greedy_mintss_lbd0(model: BaseTask):
    return greedy_mintss_with_optimizer(model, 'lbd0')


def greedy_mintss_lbd1(model: BaseTask):
    return greedy_mintss_with_optimizer(model, 'lbd1')


def greedy_mintss_lbd2(model: BaseTask):
    return greedy_mintss_with_optimizer(model, 'lbd2')


def greedy_mintss_lbd3(model: BaseTask):
    return greedy_mintss_with_optimizer(model, 'lbd3')


def greedy_mintss_lbd0s(model: BaseTask):
    return greedy_mintss_with_optimizer(model, 'lbd0s')


def greedy_mintss_lbd1s(model: BaseTask):
    return greedy_mintss_with_optimizer(model, 'lbd1')


def greedy_mintss_lbd2s(model: BaseTask):
    return greedy_mintss_with_optimizer(model, 'lbd2')


def greedy_mintss_lbd3s(model: BaseTask):
    return greedy_mintss_with_optimizer(model, 'lbd3')


def augmented_greedy_mintss_lbd0(model: BaseTask):
    return augmented_greedy_mintss_with_optimizer(model, 'lbd0')

def augmented_greedy_mintss_lbd0s(model: BaseTask):
    return augmented_greedy_mintss_with_optimizer(model, 'lbd0s')

def augmented_greedy_mintss_lbd1(model: BaseTask):
    return augmented_greedy_mintss_with_optimizer(model, 'lbd1')


def augmented_greedy_mintss_lbd2(model: BaseTask):
    return augmented_greedy_mintss_with_optimizer(model, 'lbd2')


def augmented_greedy_mintss_lbd3(model: BaseTask):
    return augmented_greedy_mintss_with_optimizer(model, 'lbd3')


def greedy_mintss_with_optimizer(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # print("gonna go1")

    opt = None

    if upb == 'lbd0':
        opt = optimizer.NormalMinOptimizer()
    elif upb == 'lbd1':
        opt = optimizer.CutoffMinOptimizer()
    elif upb == 'lbd2':
        opt = optimizer.SlicingMinOptimizer()
    elif upb == 'lbd3':
        opt = optimizer.SlicingCutoffMinOptimizer()
    elif upb == 'lbd0u':
        opt = optimizer.UnifiedSparseMinOptimizer()
    elif upb == 'lbd1u':
        opt = optimizer.UnifiedSparseMinCutoffOptimizer()
    elif upb == 'lbd2u':
        opt = optimizer.UnifiedSparseMinSlicingOptimizer()
    elif upb == 'lbd3u':
        opt = optimizer.UnifiedSparseMinSlicingCutoffOptimizer()
    elif upb == 'lbd0s':
        opt = optimizer.SievedNormalMinOptimizer()

    opt.setModel(model)
    opt.setBase([])
    opt.addIntermediate([])

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']


    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret


def greedy_mintss_lbd0u(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # eles = [model.objective(i)/model.cost_of_singleton(i) for i in model.ground_set]
    # eles.sort(reverse=True)
    #
    # ele_i = list(model.ground_set)
    # ele_i.sort(key=lambda x:model.objective(x)/model.cost_of_singleton(x), reverse=True)
    # ci = [model.cost_of_singleton(i) for i in ele_i]
    #
    # print(f"eles:{eles[:10]}, ci:{ci[:10]}")

    opt = optimizer.UnifiedSparseMinOptimizer()
    opt.setModel(model)
    opt.addIntermediate([])

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))


    opt.build()
    lambda_capital = opt.optimize()['lbd']

    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret

def greedy_mintss_lbd1u(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # print("gonna go1")

    opt = optimizer.UnifiedSparseMinCutoffOptimizer()
    opt.setModel(model)
    opt.addIntermediate([])

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']

    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret

def greedy_mintss_lbd2u(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # print("gonna go1")

    opt = optimizer.UnifiedSparseMinSlicingOptimizer()
    opt.setModel(model)
    opt.addIntermediate([])

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']

    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret

def greedy_mintss_lbd3u(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # print("gonna go1")

    opt = optimizer.UnifiedSparseMinSlicingCutoffOptimizer()
    opt.setModel(model)
    opt.addIntermediate([])

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']
    stop_time = time.time()


    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret

def greedy_mintss_opt0(model: BaseTask):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    opt = optimizer.LowerBoundOptimizer()
    opt.setModel(model)

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    opt.addIntermediate(set())

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        opt.addIntermediate(s)
        remaining_elements.remove(max_i)

    opt.build()
    lambda_capital = opt.optimize()['upb']
    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret


def augmented_greedy_mintss_with_optimizer(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # print("gonna go1")

    opt = None

    if upb == 'lbd0':
        opt = optimizer.NormalMinOptimizer()
    elif upb == 'lbd1':
        opt = optimizer.CutoffMinOptimizer()
    elif upb == 'lbd2':
        opt = optimizer.SlicingMinOptimizer()
    elif upb == 'lbd3':
        opt = optimizer.SlicingCutoffMinOptimizer()
    elif upb == 'lbd0u':
        opt = optimizer.UnifiedSparseMinOptimizer()
    elif upb == 'lbd1u':
        opt = optimizer.UnifiedSparseMinCutoffOptimizer()
    elif upb == 'lbd2u':
        opt = optimizer.UnifiedSparseMinSlicingOptimizer()
    elif upb == 'lbd3u':
        opt = optimizer.UnifiedSparseMinSlicingCutoffOptimizer()
    elif upb == 'lbd0s':
        opt = optimizer.SievedNormalMinOptimizer()
    elif upb == 'lbd1s':
        opt = optimizer.UnifiedSparseMinCutoffOptimizer()
    elif upb == 'lbd2s':
        opt = optimizer.UnifiedSparseMinSlicingOptimizer()
    elif upb == 'lbd3s':
        opt = optimizer.UnifiedSparseMinSlicingCutoffOptimizer()

    opt.setModel(model)
    opt.setBase([])
    opt.addIntermediate([])

    s = set()

    augmented_s, augmented_c = None, 0

    while g(s) < model.value:
        # augment procedure
        for i in remaining_elements:
            temp_s = s | {i}
            v = model.objective(list(temp_s))
            if v >= model.value:
                print(f"i am here, v:{v}, s:{s}, temp_s:{temp_s}, cost")
                if augmented_s is None or model.cost_of_set(temp_s) < augmented_c:
                    augmented_s = temp_s
                    augmented_c = model.cost_of_set(temp_s)


        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")
        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']

    if augmented_c < model.cost_of_set(list(s)):
        print("agu")
        s = augmented_s

    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret

def augmented_greedy_mintss_lbd0u(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # eles = [model.objective(i)/model.cost_of_singleton(i) for i in model.ground_set]
    # eles.sort(reverse=True)
    #
    # ele_i = list(model.ground_set)
    # ele_i.sort(key=lambda x:model.objective(x)/model.cost_of_singleton(x), reverse=True)
    # ci = [model.cost_of_singleton(i) for i in ele_i]
    #
    # print(f"eles:{eles[:10]}, ci:{ci[:10]}")

    opt = optimizer.UnifiedSparseMinOptimizer()
    opt.setModel(model)
    opt.addIntermediate([])

    s = set()

    augmented_s, augmented_c = None, 0

    while g(s) < model.value:
        # augment procedure
        for i in remaining_elements:
            temp_s = s | {i}
            v = model.objective(list(temp_s))
            if v >= model.value:
                print(f"i am here, v:{v}, s:{s}, temp_s:{temp_s}")

                if augmented_s is None or model.cost_of_set(temp_s) < augmented_c:
                    augmented_s = temp_s
                    augmented_c = model.cost_of_set(temp_s)

        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']

    if augmented_c < model.cost_of_set(list(s)):
        s = augmented_s

    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret

def augmented_greedy_mintss_lbd1u(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # print("gonna go1")

    opt = optimizer.UnifiedSparseMinCutoffOptimizer()
    opt.setModel(model)
    opt.addIntermediate([])

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']

    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret


def augmented_greedy_mintss_lbd2u(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # print("gonna go1")

    opt = optimizer.UnifiedSparseMinSlicingOptimizer()
    opt.setModel(model)
    opt.addIntermediate([])

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']

    stop_time = time.time()

    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret

def augmented_greedy_mintss_lbd3u(model: BaseTask, upb=None):
    start_time = time.time()
    parameters = {}
    lambda_capital = 0

    def g(x):
        return min(model.objective(list(x)), model.value)

    def g_s(x, base):
        item1 = set(base) | {x}
        item2 = base

        return g(item1) - g(item2)

    def d_g_s(x, base):
        return g_s(x, base) / model.cost_of_singleton(x)

    remaining_elements = set(model.ground_set)

    # print("gonna go")
    # print("gonna go1")

    opt = optimizer.UnifiedSparseMinSlicingCutoffOptimizer()
    opt.setModel(model)
    opt.addIntermediate([])

    s = set()
    while g(s) < model.value:
        max_i, max_d = None, 0
        for i in remaining_elements:
            if max_i is None or d_g_s(i, s) > max_d:
                max_i = i
                max_d = g_s(i, s)

        s.add(max_i)
        remaining_elements.remove(max_i)

        # print(f"s updated:{s}")

        opt.addIntermediate(copy.deepcopy(list(s)))

    opt.build()
    lambda_capital = opt.optimize()['lbd']
    stop_time = time.time()


    ret = {
        "S": s,
        "f(S)": model.objective(list(s)),
        "c(S)": model.cost_of_set(list(s)),
        "upb": lambda_capital,
        "AF": model.cost_of_set(list(s)) / lambda_capital,
        "time": stop_time - start_time
    }

    return ret