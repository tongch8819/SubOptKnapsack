import argparse
import os
import pickle
import random

import numpy as np

import a_star
import data_correcting
import filter_search
import id_aster
import model_factory
from a_star import Astar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default='', help="task name")
    parser.add_argument("-n", '--num', default=100, help='size of the ground set')
    parser.add_argument("-a", "--archive", default=27, help="archive index")
    parser.add_argument("-hf", "--heuristic", default='ub0', help="the heuristic function")
    parser.add_argument("-aa", "--alpha", default=0.8, help="the approximation factor")
    parser.add_argument("-g", "--algorithm", default='FS', help="the searching algorithm")
    parser.add_argument("-d", "--sorting", default='g', help="the sorting function for breaking ties")
    args = parser.parse_args()

    assert args.heuristic in ['ub0', 'ub1', 'ub2', 'ub0+', 'ub1+', 'ub2+', 'ub4']

    # ub_list = [args.heuristic]

    ub_list = ['ub0', 'ub2']
    d_list = ['d']

    alpha = float(args.alpha)

    start_seed = 0
    stop_seed = 1

    interval = 1
    num_points = 6
    start_point = 10
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    root_dir = os.path.join("./result", f"archive-{args.archive}")

    for seed in range(start_seed, stop_seed):
        for budget in bds:
            for ub in ub_list:
                for d in d_list:
                    random.seed(seed)
                    model = model_factory.model_factory(args.task, int(args.num), seed, budget, knap=True)

                    alg = None
                    if args.algorithm == 'FS':
                        alg = filter_search.FS(model)
                    elif args.algorithm == 'AFS':
                        alg = filter_search.AugmentedFS(model)
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'BAFSa':
                        alg = filter_search.BestAugmentedFS(model)
                        alg.use_alpha = True
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'BAFSna':
                        alg = filter_search.BestAugmentedFS(model)
                        alg.use_alpha = False
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'IDA':
                        alg = id_aster.IDAstar(model)
                    elif args.algorithm == 'Astar':
                        alg = a_star.Astar(model)
                        alg.set_h(heuristic=ub)

                    alg.alpha = alpha
                    alg.setOpt(ub)
                    alg.build()
                    res = alg.optimize()
                    print(f"Done:seed:{seed}/{stop_seed - start_seed + 1}, ub:{ub}, d:{d}, budget:{budget}, res:{res}")

                    save_dir = os.path.join(root_dir, args.task, f'{args.num}', f'{seed}')
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)

                    save_path = os.path.join(save_dir, "{}-{}-{}-{}-{}-{}.pckl".format(
                        args.algorithm, ub, d, budget, alpha, model.__class__.__name__))

                    with open(save_path, "wb") as wrt:
                        pickle.dump(res, wrt)