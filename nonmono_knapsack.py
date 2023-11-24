from base_task import BaseTask

import math
import random
from typing import Set, List
from copy import deepcopy
import bisect


def argmax(domain, func, with_fv=False):
    v, fv = None, None
    for e in domain:
        new_fv = func(e)
        if fv is None or fv < new_fv:
            v, fv = e, new_fv
    if with_fv:
        return v, fv
    else:
        return v


def PARKNAPSACK(model: BaseTask, epsilon: float = None, alpha: float = None, p: float = None):
    """
    Inputs:
    - epsilon: precision
    - alpha: parameter, dafault is 2 - sqrt(3)
    - p: sampling probability
    """
    if epsilon is None:
        epsilon = 0.25
    assert epsilon < 1 / 3
    if alpha is None:
        alpha = 2 - math.sqrt(3)
    if p is None:
        p = (1 - alpha) / 2
    assert 0 <= p <= 1.0
    N = model.ground_set
    f = model.objective
    B = model.budget
    n = len(N)
    N_minus, N_plus = [], []
    small_element_cost_threhold = B / n
    for e in N:
        cx = model.cost_of_singleton(e)
        if cx < small_element_cost_threhold:
            N_minus.append(e)
        else:
            N_plus.append(e)

    # change int into list, containing single element
    x_star = argmax(N_plus, lambda x: f([x]))
    tau_hat = alpha * n * f(x_star) / B
    epsilon_hat = epsilon / 125
    l = epsilon_hat * epsilon_hat
    epsilon_hat_inverse = epsilon_hat
    k = epsilon_hat_inverse * math.log(n)
    S_minus = SUBMODMAX(N_minus, epsilon_hat)

    H = []
    for e in N_plus:
        if random.random() <= p:
            H.append(e)

    precompute_tau = []
    scale = 1 - epsilon_hat
    cur = tau_hat
    for _ in range(k + 1):
        precompute_tau.append(cur)
        cur *= scale

    T, f_T = argmax([[x_star], S_minus], f, with_fv=True)

    # TODO: change it into parallel computing
    for i in range(k + 1):
        tau_i = precompute_tau[i]
        upb = math.ceil(epsilon_hat_inverse * math.log(epsilon_hat_inverse))
        for j in range(1, upb + 1):
            S_ij = THRESHSEQ(H, tau_i, epsilon_hat, l, B, model)
            f_S_ij = f(S_ij)
            if f_S_ij > f_T:
                T, f_T = S_ij, f_S_ij
    return T


def SUBMODMAX():
    # TODO: assume access to unconstrained submodular maximization with epsilon AF and adaptivity
    pass


def THRESHSEQ(raw_X: List[int], tau: float, epsilon: float, l: float, B: float, model: BaseTask) -> Set[int]:
    """
    Inputs:
    - raw_X: set of elements
    - tau: threshold > 0
    - epsilon: precision, it lies in [0, 1]
    - l: parameter
    - B: budget
    - model: problem instance

    Returns:
    - a solution set of elements
    """
    assert tau > 0
    assert 0 <= epsilon <= 1
    S = set()
    ctr = 0
    f = model.objective

    X = set()
    for a in raw_X:
        if f([a]) >= tau * model.cost_of_singleton(a):
            X.add(a)

    while len(X) and ctr < l:
        a_seq = SAMPLESEQ(S, X, B)
        A_seq = [{}]  # list of set
        cur_set = {}
        for a in a_seq:
            cur_set.add(a)
            A_seq.append(deepcopy(cur_set))

        d = len(a_seq)
        G_seq = [[]] + [None] * d  # list of list
        E_seq = [[]] + [None] * d  # list of list
        for i in range(1, d + 1):
            A_i = A_seq[i]
            S_union_A_i = S | A_i
            X_minus_A_i = X - A_i
            X_i = []
            G_i, E_i = [], []
            for a in X_minus_A_i:
                cost_of_a = model.cost_of_singleton(a)
                marginal_gain_a_of_S_union_A_i = model.marginal_gain(
                    a, S_union_A_i)
                if cost_of_a + model.cost_of_set(S_union_A_i) <= B:
                    X_i.append(a)
                    # inside X_i, we find elements for G_i and E_i
                    if marginal_gain_a_of_S_union_A_i >= tau * cost_of_a:
                        G_i.append(a)
                    if marginal_gain_a_of_S_union_A_i < 0:
                        E_i.append(a)

            G_seq[i] = G_i
            E_seq[i] = E_i

        idxs = [i for i in range(1, d + 1)]
        offset = (1 - epsilon) * model.cost_of_set(X)
        i_star = bisect.bisect_right(
            idxs, x=0., key=lambda x: model.cost_of_set(G_seq[x]) - offset)

        def j_star_cmp_func(j: int):
            A_j = A_seq[j]
            S_union_A_j = S | A_j
            sum_over_G, sum_over_E = 0., 0.
            G_j = G_seq[j]
            for x in G_j:
                sum_over_G += model.marginal_gain(x, S_union_A_j)
            sum_over_G *= epsilon
            E_j = E_seq[j]
            for x in E_j:
                sum_over_E += abs(model.marginal_gain(x, S_union_A_j))
            return sum_over_G - sum_over_E

        j_star = bisect.bisect_right(idxs, x=0., key=j_star_cmp_func)
        k_star = min(i_star, j_star)
        S = S | A_seq[k_star]
        X = G_seq[k_star]
        if j_star < i_star:
            ctr += 1

    return S


def SAMPLESEQ(S: List[int], raw_X: Set[int], B: float, model: BaseTask) -> List[int]:
    """
    Inputs:
    - S: current solution
    - X: set of remaining elements
    - B: budget

    Returns:
    - a list of elements
    """
    assert B > 0
    assert type(raw_X) is set
    A = []
    i = 1
    cost_A = 0.
    cost_S = model.cost_of_set(S)
    X = deepcopy(raw_X)
    while len(X):
        a_i = random.choice(X)
        A.append(a_i)
        cost_A += model.cost_of_singleton(a_i)
        X.remove(a_i)
        to_remove_set = {}
        for x in X:
            if model.cost_of_singleton(x) + cost_A + cost_S > B:
                to_remove_set.add(x)
        X = X - to_remove_set
        i += 1

    return A
