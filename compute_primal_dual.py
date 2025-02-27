import argparse
import os
import pickle
import random
import time

import numpy as np

import mgreedy
import model_factory
from optimizer import PrimalDualOptimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default='', help="task name")
    parser.add_argument("-n", '--num', default=1000, help='size of the ground set')
    parser.add_argument("-a", "--archive", default=27, help="archive index")
    args = parser.parse_args()

    alg = PrimalDualOptimizer()

    start_seed = 0
    stop_seed = 20

    interval = 1
    num_points = 15
    start_point = 6
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    root_dir = os.path.join("./result", f"archive-{args.archive}")

    for seed in range(start_seed, stop_seed):
        for budget in bds:
            random.seed(seed)
            model = model_factory.model_factory(args.task, int(args.num), seed, budget, knap=True)

            start_time = time.time()

            alg.model = model
            alg.build()
            alg_output = alg.optimize()

            mgreedy_output = mgreedy.modified_greedy_plain(model)

            S= mgreedy_output['S']
            value = mgreedy_output['f(S)']

            stop_time = time.time()

            assert  value / alg_output['upb'] <= 1.0

            res = {
                "S": S,
                "f(S)": value,
                "fx": alg_output['fx'],
                "upb": alg_output['upb'],
                "AF": value / alg_output['upb'],
                "time": stop_time - start_time
            }

            print(f"Done:seed:{seed}/{stop_seed - start_seed + 1}, budget:{budget}, res:{res}")

            save_dir = os.path.join(root_dir, args.task, f'{args.num}', f'{seed}')
            save_path = os.path.join(save_dir, "{}-{}-{}.pckl".format(
                "modified_greedy", 'PMD' ,budget, model.__class__.__name__))

            with open(save_path, "wb") as wrt:
                pickle.dump(res, wrt)

