import os
import pickle

import greedy_min
from compute_knapsack_exp import model_factory
import numpy as np

def compute_min_series(task):
    seed_start = 0
    seed_end = 100
    n = 500
    root_dir = f"./result/archive_min"
    start_value = 0.5
    num_points = 10
    interval = 0.5
    end_value = start_value + (num_points - 1) * interval
    values = np.linspace(start=start_value, stop=end_value, num=num_points)

    upb = 'ub2'

    for seed in range(seed_start, seed_end):
        model = model_factory(task, n, seed, 0)
        save_dir = os.path.join(root_dir, "youtube", f"{n}", f"{seed}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for value in values:
            model.value = value
            res = greedy_min.simple_greedy_min(model, upb)

            save_path = os.path.join(save_dir, "{}-{}-{:.2f}-{}.pckl".format(upb, model.__class__.__name__, value, seed))
            with open(save_path, "wb") as wrt:
                pickle.dump(res, wrt)
            print(f"seed:{seed}/{seed_end}, value:{value}/{end_value}")
            print(res)

    pass

if __name__ == "__main__":
    compute_min_series("facebook")
    pass