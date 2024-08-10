from nonmono_data_dependent_upperbound import singleton_knapsack_fill

from base_task import BaseTask
from typing import Callable, Iterable
from functools import partial
import itertools
from copy import deepcopy
import multiprocessing

def find_subsets(s: Iterable, n: int):
    return list(itertools.combinations(s, n))

def filter_invalid_singleton(model: BaseTask):
    # filter invalid singleton from ground set of model inplace
    V = model.ground_set
    V = list(filter(lambda e: model.cost_of_singleton(e) <= model.budget, V))
    model.ground_set = V


def set_enumeration(model: BaseTask, alg_handler: Callable, num_initial_elements: int = 2, upb: str = None, verbose = False):
    n = len(model.ground_set)
    assert num_initial_elements > 0 and num_initial_elements <= n

    # filter out invalid elements in O(n)
    filter_invalid_singleton(model)

    base_set_lst = find_subsets(model.ground_set, num_initial_elements)
    optimal_base_set, residual_res = None, None
    assert len(base_set_lst) > 0, "Empty base set list. Abort"

    if verbose:
        print("Total numebr of base set:", len(base_set_lst))
        print("Size of V:", n)
    for base_set in base_set_lst:
        base_total_cost = model.cost_of_set(base_set)
        if verbose:
            print("c(Y) and b", base_total_cost, model.budget)
        if base_total_cost > model.budget:
            # enumeration violates the cost constraint
            continue

        # construct residual instance
        modified_model = deepcopy(model)
        # C1 = model.objective(base_set)  
        # assume S and base set are not overlapping with each other
        modified_model.objective = lambda S: model.objective(list(S) + list(base_set))
        modified_model.budget = model.budget - base_total_cost
        # assert budget of residual instance is no less than zero
        assert modified_model.budget >= 0.
        modified_model.ground_set = list(set(model.ground_set) - set(base_set))
        filter_invalid_singleton(modified_model)

        # solve residual instance
        cur_residual_res = alg_handler(modified_model, upb = None)
        if verbose:
            print("Current base:", base_set)
            print("Current sub solution:", cur_residual_res)
        if residual_res is None or residual_res['f(S)'] < cur_residual_res['f(S)']:
            residual_res = cur_residual_res
            optimal_base_set = base_set

    # combine solution of residual instance with base set
    assert optimal_base_set is not None
    res = {
        'S' : residual_res['S'].union(set(optimal_base_set)),
    }
    res['f(S)'] = model.objective(list(res['S']))
    res['c(S)'] = model.cost_of_set(base_set) + residual_res['c(S)']
    if upb is not None:
        res['Upb'] = singleton_knapsack_fill(model)
        res['AF'] = res['f(S)'] / res['Upb']
    return res


def start_with_base_set(base_set, model, alg_handler, res, verbose=False):
    base_total_cost = model.cost_of_set(base_set)
    if verbose:
        print("c(Y) and b", base_total_cost, model.budget)
    if base_total_cost > model.budget:
        # enumeration violates the cost constraint
        # early stop
        return
    # construct residual instance
    modified_model = deepcopy(model)
    # C1 = model.objective(base_set)  
    # assume S and base set are not overlapping with each other
    modified_model.objective = lambda S: model.objective(list(S) + list(base_set))
    modified_model.budget = model.budget - base_total_cost
    # assert budget of residual instance is no less than zero
    assert modified_model.budget >= 0.
    modified_model.ground_set = list(set(model.ground_set) - set(base_set))
    filter_invalid_singleton(modified_model)

    # solve residual instance
    cur_residual_res = alg_handler(modified_model, upb = None)

    res[tuple(base_set)] = cur_residual_res



def set_enumeration_parallel(model: BaseTask, alg_handler: Callable, num_initial_elements: int = 2, upb: str = None, verbose = False):
    n = len(model.ground_set)
    assert num_initial_elements > 0 and num_initial_elements <= n

    # filter out invalid elements in O(n)
    filter_invalid_singleton(model)

    base_set_lst = find_subsets(model.ground_set, num_initial_elements)
    assert len(base_set_lst) > 0, "Empty base set list. Abort"

    if verbose:
        print("Total numebr of base set:", len(base_set_lst))
        print("Size of V:", n)
    
    # create a shared variable to communicate 
    # reference: https://stackoverflow.com/questions/10415028/how-can-i-get-the-return-value-of-a-function-passed-to-multiprocessing-process
    manager = multiprocessing.Manager()
    res = manager.dict()

    jobs = []
    for base_set in base_set_lst:
        p = multiprocessing.Process(target=start_with_base_set, args=(base_set, model, alg_handler, res))
        jobs.append(p)
        p.start()

    # wait for every process to be finished
    for proc in jobs:
        proc.join()
    
    # print(res)
    optimal_base_set, residual_res = None, None
    for base_set, cur_residual_res in res.items():
        # base_set is tuple in order to become hashable
        if residual_res is None or residual_res['f(S)'] < cur_residual_res['f(S)']:
            residual_res = cur_residual_res
            optimal_base_set = list(base_set)

    # combine solution of residual instance with base set
    assert optimal_base_set is not None
    res = {
        'S' : set(residual_res['S']).union(set(optimal_base_set)),
    }
    res['f(S)'] = model.objective(list(res['S']))
    res['c(S)'] = model.cost_of_set(base_set) + residual_res['c(S)']
    if upb is not None:
        res['Upb'] = singleton_knapsack_fill(model)
        res['AF'] = res['f(S)'] / res['Upb']
    return res

# def three_set_enumeration(model: BaseTask, alg_handler: Callable, upb: str = None):
#     return set_enumeration(model, 3, alg_handler, upb)

# def two_set_enumeration(model: BaseTask, alg_handler: Callable, upb: str = None):
#     return set_enumeration(model, 2, alg_handler, upb)

# three_set_enumeration = partial(set_enumeration, num_initial_elements=3)
# two_set_enumeration = partial(set_enumeration, num_initial_elements=2)
