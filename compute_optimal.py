import argparse
import os
import pickle
import random

import numpy as np

import a_star
import filter_search
import model_factory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default='', help="task name")
    parser.add_argument("-n", '--num', default=25, help='size of the ground set')
    parser.add_argument("-a", "--archive", default=18, help="archive index")
    parser.add_argument("-hf", "--heuristic", default='ub0', help="the heuristic function")
    args = parser.parse_args()

    # assert args.heuristic in ['ub0', 'ub2', 'ub3']

    ub_list = [args.heuristic]

    start_seed = 0
    stop_seed = 1

    interval = 1
    num_points = 35
    start_point = 6
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    root_dir = os.path.join("./result", f"archive-{args.archive}")

    for seed in range(start_seed, stop_seed):
        for budget in bds:
            for ub in ub_list:
                random.seed(seed)
                model = model_factory.model_factory(args.task, int(args.num), seed, budget, knap=True)
                alg = a_star.A_star(model)
                alg.setOpt(ub)
                alg.build()
                res = alg.optimize()
                print(f"Done:seed:{seed}/{stop_seed - start_seed + 1}, budget:{budget}, res:{res}")

                save_dir = os.path.join(root_dir, args.task, f'{args.num}')
                save_path = os.path.join(save_dir, "{}-{}-{}-{}.pckl".format(
                    "Astar", ub, seed, model.__class__.__name__))

                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
