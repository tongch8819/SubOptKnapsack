"""
This script implements data-dependent upperbound for CC problem.
"""

from base_task import BaseTask

from typing import Set, List
from itertools import accumulate
import bisect
import numpy as np
from copy import deepcopy


def find_partition_sum(model: BaseTask, A: List, f_a: List):
    """
    Method 3 implementation by appendix of original paper.

    Inputs:
    - model: problem instance
    - A: ground set sorted in descending order of singleton function value
    - f_a: sorted singleton function value in descending order
    """
    f = model.objective
    k = model.budget
    n = len(model.ground_set)

    # fa = [f([x]) for x in model.ground_set]
    # A = deepcopy(model.ground_set)
    # A.sort(key=lambda x: fa[x], reverse=True)  # decreasing singleton value
    cur_sum = 0.
    for j in range(1, k + 1):
        # binary search 
        left, right = 0, n - 1
        while left < right:
            mid = (left + right) // 2
            i = mid
            # f(a_i) is A[i - 1]
            f_a_i = f_a[i]
            # f(A_i) is objective( A[:i] )
            f_A_i = f(A[:i+1])
            val = f_A_i - cur_sum - f_a_i
            if val >= 0:
                right = mid - 1
            else:
                left = mid + 1
        i_star = left

        # i_star -> A[: i_star + 1]
        # i_star - 1 -> A[: i_star]
        assert i_star >= 0
        f_A_i_star_minus_1 = f( A[: i_star])
        f_a_i_star = f_a[i_star]
        if f_A_i_star_minus_1 - cur_sum - f_a_i_star >= 0:
            i_star -= 1
            assert i_star + 1 >= 0
            f_A_i_star = f( A[: i_star + 1] )
            v_j = f_A_i_star - cur_sum
        else:
            v_j = f_a_i_star
        
        cur_sum += v_j
    
    return cur_sum


def dual(model: BaseTask, guess_collection: List[List[int]]):
    f = model.objective
    N = model.ground_set

    # precompute the prefix sequence, decreasing singleton value
    t = sorted([(x, f([x])) for x in model.ground_set], key=lambda y: y[1], reverse=True)
    A, f_a = list(zip(*t))

    opt_bar = f(N)
    for S in guess_collection:
        opt_bar_prime = find_partition_sum(model, A, f_a)
        opt_bar = min(f(S) + opt_bar_prime, opt_bar)
    return opt_bar_prime