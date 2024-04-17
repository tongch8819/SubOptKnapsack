from base_task import BaseTask

from data_dependent_upperbound import marginal_delta, marginal_delta_version4
from data_dependent_upperbound import marginal_delta_version2
from data_dependent_upperbound import marginal_delta_version3
from data_dependent_upperbound import marginal_delta_for_streaming_version1, marginal_delta_for_streaming_version3
from data_dependent_upperbound import marginal_delta_for_knapsack_streaming_version1

import numpy as np
import time


def sieve_streaming(model: BaseTask, upb: str = 'ub0', epsilon=0.2):
    start_time = time.time()

    alpha = 0.5

    sol_v = []

    max_val = float('-inf')

    upb_val = float('-inf')

    for i in list(model.ground_set):
        if model.objective([i]) > max_val:
            max_val = model.objective([i])

    if upb == 'ub0':
        upb_val = max_val * model.budget
    elif upb == 'ub1':
        upb_val = marginal_delta_for_streaming_version1(set(), set(model.ground_set), model)
    elif upb == 'ub3':
        upb_val = marginal_delta_for_streaming_version3(set(), set(model.ground_set), model)

    O = []
    threshold = max_val
    factor = 1 + epsilon

    while threshold < upb_val:
        if threshold >= max_val:
            O.append(threshold)
        threshold *= factor

    for j in range(0, len(O)):
        sol_v.append(set())

    print(f"O size:{len(O)}")
    for i in list(model.ground_set):
        for j in range(0, len(O)):
            if len(sol_v[j]) < model.budget \
                    and model.marginal_gain(i, list(sol_v[j])) >= (O[j] * alpha - model.objective(sol_v[j])) / (
                    model.budget - len(sol_v[j])):
                sol_v[j] = sol_v[j] | {i}

    sol = max(sol_v, key=lambda x: model.objective(list(x)))

    stop_time = time.time()

    res = {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': model.cost_of_set(sol),
        'Lambda': upb_val,
        'Time': stop_time - start_time
    }

    return res


def sieve_one_pass_streaming(model: BaseTask, epsilon=0.2):
    start_time = time.time()

    alpha = 0.9

    sol_v = {}

    max_val = float('-inf')

    threshold = -1
    factor = 1 + epsilon

    for i in list(model.ground_set):
        if model.objective([i]) > max_val:
            max_val = model.objective([i])
            if threshold == -1:
                threshold = max_val
            upb_val = max_val * model.budget / alpha

            while threshold < upb_val:
                if threshold >= max_val:
                    sol_v[threshold] = []
                threshold *= factor

            to_remove = set()
            for sv_key in sol_v.keys():
                if sv_key < max_val:
                    to_remove.add(sv_key)

            for sv_key in to_remove:
                sol_v.pop(sv_key)

        for sv_key in sol_v.keys():
            if len(sol_v[sv_key]) < model.budget \
                    and model.marginal_gain(i, sol_v[sv_key]) >= (sv_key * alpha - model.objective(sol_v[sv_key])) / (
                    model.budget - len(sol_v[sv_key])):
                sol_v[sv_key].append(i)

    sol = max([sol_v[sv_key] for sv_key in sol_v.keys()], key=lambda x: model.objective(x))

    stop_time = time.time()

    res = {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': model.cost_of_set(sol),
        'Lambda': -1,
        'Time': stop_time - start_time
    }

    return res


def sieve_streaming_ub0(model: BaseTask):
    return sieve_streaming(model)


def sieve_streaming_ub1(model: BaseTask):
    return sieve_streaming(model, "ub1")


def sieve_streaming_ub2(model: BaseTask):
    return sieve_streaming(model, "ub2")


def sieve_streaming_ub3(model: BaseTask):
    return sieve_streaming(model, "ub3")


def sieve_streaming_ub4(model: BaseTask):
    return sieve_streaming(model, "ub4")


def sieve_knapsack_streaming(model: BaseTask, upb: str = 'ub0', epsilon=0.2):
    start_time = time.time()

    alpha = 0.9

    sol_v = []

    max_val = float('-inf')
    max_density = float('-inf')
    max_ele = -1

    upb_val = float('-inf')

    for i in list(model.ground_set):
        if model.density(i, []) > max_density and model.cost_of_singleton(i) <= model.budget:
            max_density = model.density(i, [])
        if model.objective([i]) > max_val and model.cost_of_singleton(i) <= model.budget:
            max_val = model.objective([i])
            max_ele = i

    if upb == 'ub0':
        upb_val = max_density * model.budget
    elif upb == 'ub1':
        upb_val = marginal_delta_for_knapsack_streaming_version1(set(), set(model.ground_set), model)

    O = []
    threshold = max_val
    factor = 1 + epsilon

    candidate_count = 0

    while threshold < upb_val:
        # print(f"current threshold:{threshold}, upb_val:{upb_val}, candidate count:{candidate_count}")
        if threshold >= max_val:
            O.append(threshold)
        threshold *= factor
        candidate_count += 1

    for j in range(0, len(O)):
        sol_v.append(set())

    print(f"O size:{len(O)}")
    for i in list(model.ground_set):
        for j in range(0, len(O)):
            if model.cost_of_set(sol_v[j]) + model.cost_of_singleton(i) < model.budget \
                    and model.marginal_gain(i, list(sol_v[j])) >= (O[j] * alpha - model.objective(sol_v[j])) / (
                    model.budget - model.cost_of_set(sol_v[j])):
                sol_v[j] = sol_v[j] | {i}

    sol = max(sol_v, key=lambda x: model.objective(list(x)))

    if model.objective(sol) < max_val:
        sol = [max_ele]

    stop_time = time.time()

    res = {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': model.cost_of_set(sol),
        'Lambda': upb_val,
        'Time': stop_time - start_time,
    }

    return res


def sieve_knapsack_streaming_ub0(model: BaseTask):
    return sieve_knapsack_streaming(model)


def sieve_knapsack_streaming_ub1(model: BaseTask):
    return sieve_knapsack_streaming(model, "ub1")


def sieve_knapsack_streaming_ub2(model: BaseTask):
    return sieve_knapsack_streaming(model, "ub2")


def sieve_knapsack_streaming_ub3(model: BaseTask):
    return sieve_knapsack_streaming(model, "ub3")


def sieve_knapsack_streaming_ub4(model: BaseTask):
    return sieve_knapsack_streaming(model, "ub4")

def sieve_one_pass_streaming_ub0(model: BaseTask):
    return sieve_one_pass_streaming(model)


def sieve_one_pass_streaming_ub1(model: BaseTask):
    return sieve_one_pass_streaming(model)


def sieve_one_pass_streaming_ub2(model: BaseTask):
    return sieve_one_pass_streaming(model)


def sieve_one_pass_streaming_ub3(model: BaseTask):
    return sieve_one_pass_streaming(model)


def sieve_one_pass_streaming_ub4(model: BaseTask):
    return sieve_one_pass_streaming(model)
