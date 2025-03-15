import argparse
import copy
import math
import os
import pickle

import greedy_min
from compute_knapsack_exp import model_factory
import numpy as np


def compute_min_series(task):
    seed_start = 0
    seed_end = 100
    n = 1000
    root_dir = f"./result/archive-29"

    upb = 'ub0'

    for seed in range(seed_start, seed_end):
        model = model_factory(task, n, seed, budget=0, knap=False)

        t = copy.deepcopy(model.ground_set)
        t.sort(key=lambda x: model.cost_of_singleton(x))
        gate = t[0] + t[1]
        max_v = 0
        max_s = None
        for ele in t:
            t_c = model.cost_of_singleton(ele)
            t_v = model.objective(ele)
            if t_c < gate and (max_s is None or max_v < t_v):
                max_s = ele
                max_v = t_v

        start_value = min(max_v + 10, 50)
        num_points = 15
        interval = 5
        end_value = start_value + (num_points - 1) * interval
        values = np.linspace(start=start_value, stop=end_value, num=num_points)

        save_dir = os.path.join(root_dir, task, f"{n}", f"{seed}")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for value in values:
            model.value = value
            res = greedy_min.simple_greedy_min(model, upb)
            res['ground'] = n
            res['worst'] = 1 + math.log(value, math.e)
            res['start_v'] = start_value

            save_path = os.path.join(save_dir,
                                     "{}-{}-{:.2f}-{}.pckl".format(upb, model.__class__.__name__, value, seed))
            with open(save_path, "wb") as wrt:
                pickle.dump(res, wrt)
            print(res)
    pass


def compute_min_series_integer(task, knap=True, archive=29, upb='ub0'):
    seed_start = 0
    seed_end = 200
    n = 1000
    root_dir = f"./result/archive-{archive}"

    for seed in range(seed_start, seed_end):
        model = model_factory(task, n, seed, budget=0, knap=knap)

        # num_points = 10
        # start_value = 10
        # interval = 10

        num_points = 10
        start_value = int(n/4)
        interval = int(n/20)

        # num_points = 1
        # start_value = 550
        # interval = 10

        end_value = start_value + (num_points - 1) * interval
        values = np.linspace(start=start_value, stop=end_value, num=num_points)

        save_dir = os.path.join(root_dir, task, f"{n}", f"{seed}")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for value in values:
            model.value = value
            res = greedy_min.greedy_mintss(model, upb)
            res['ground'] = n

            max_ele, max_v = None, -1
            for ele in model.ground_set:
                if max_ele is None or model.objective([ele]) > max_v:
                    max_v = model.objective([ele])
                    max_ele = ele

            res['worst'] = 1 + math.log(max_v, math.e)

            save_path = os.path.join(save_dir,
                                     "{}-{}-{:.2f}-{}.pckl".format(upb, model.__class__.__name__, value, seed))
            with open(save_path, "wb") as wrt:
                pickle.dump(res, wrt)

            print(f"Done:seed:{seed}, value:{value}")
            print(res)
    pass


def compute_min_series_b(task):
    seed_start = 0
    seed_end = 10
    root_dir = f"./result/archive-29"

    upb = 'ub2'

    for n in range(100, 501, 100):
        n_dir = os.path.join(root_dir, task, f"{n}")
        if not os.path.exists(n_dir):
            os.mkdir(n_dir)

        for seed in range(seed_start, seed_end):
            model = model_factory(task, n, seed, budget=0, knap=False)
            # calculate worst case
            max_ele, max_v = None, -1
            for ele in model.ground_set:
                if max_ele is None or model.objective([ele]) > max_v:
                    max_v = model.objective([ele])
                    max_ele = ele

            save_dir = os.path.join(n_dir, f'{seed}')

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            model.value = n
            print(f"n:{n}, seed:{seed}")
            res = greedy_min.simple_greedy_min(model, upb)

            res['ground'] = n
            res['worst'] = 1 + math.log(max_v, math.e)

            # assert res['AF'] <= res['worst'], print(f"AF:{res['AF']}, worst:{res['worst']}")

            save_path = os.path.join(save_dir, "{}-{}-{:.2f}-{}.pckl".format(upb, model.__class__.__name__, n, seed))
            with open(save_path, "wb") as wrt:
                pickle.dump(res, wrt)
            print(res)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default='', help="task name")
    parser.add_argument("-b", "--budget", default=True, help="use budget function")
    parser.add_argument("-a", "--archive", default=29, help="archive")
    parser.add_argument("-k", "--knapsack", default=True, help="knapsack")
    parser.add_argument("-u", "--upb", default=True, help="upper bound")
    args = parser.parse_args()

    assert args.task in ["facebook", "youtube"]

    compute_min_series_integer(task=args.task, knap=args.knapsack, archive=args.archive, upb=args.upb)
