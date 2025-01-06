import copy
import math
import os
import pickle

import greedy_min
from compute_knapsack_exp import model_factory
import numpy as np


def compute_min_series(task):
    seed_start = 0
    seed_end = 10
    n = 200
    root_dir = f"./result/archive-21"

    upb = 'ub0'

    for seed in range(seed_start, seed_end):
        model = model_factory(task, n, seed, 0)

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
        print(f"seed:{seed}, max_v:{max_v}, max_s:{max_s}")
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

            save_path = os.path.join(save_dir, "{}-{}-{:.2f}-{}.pckl".format(upb, model.__class__.__name__, value, seed))
            with open(save_path, "wb") as wrt:
                pickle.dump(res, wrt)
            print(res)
    pass


if __name__ == "__main__":
    compute_min_series("youtube")
    pass
