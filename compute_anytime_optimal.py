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
    parser.add_argument("-aa", "--alpha", default=1.0, help="the approximation factor")
    parser.add_argument("-g", "--algorithm", default='FS', help="the searching algorithm")
    parser.add_argument("-d", "--sorting", default='g', help="the sorting function for breaking ties")
    parser.add_argument("-rt", "--runningtime", default=1000, help="the running time limitation")
    args = parser.parse_args()

    assert args.heuristic in ['ub0', 'ub1', 'ub2', 'ub0+', 'ub1+', 'ub2+', 'ub4', 'dom']

    # ub_list = [args.heuristic]

    ub_list = ['ub2']
    d_list = ['d']

    alpha = float(args.alpha)
    running_time = float(args.runningtime)

    start_seed = 0
    stop_seed = 1

    interval = 1
    num_points = 10
    start_point = 26
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
                    elif args.algorithm == 'BAFSmore':
                        alg = filter_search.BestAugmentedMoreFS(model)
                        alg.use_alpha = False
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'BAFSmorealpha':
                        alg = filter_search.BestAugmentedMoreFS(model)
                        alg.use_alpha = True
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'BAFSmorenpb':
                        alg = filter_search.BestAugmentedMoreFS(model)
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'IDA':
                        alg = id_aster.IDAstar(model)
                    elif args.algorithm == 'Astar':
                        alg = a_star.Astar(model)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'Efficient':
                        alg = filter_search.EfficientBranchAndBound(model)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'BasicEfficient':
                        alg = filter_search.EfficientBranchAndBound(model)
                        alg.basic_mode = True
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'EfficientBFS':
                        alg = filter_search.EfficientBFS(model)
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'EfficientBFSNheap':
                        alg = filter_search.EfficientBFS(model)
                        alg.heap_class = 'simple'
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'EfficientBFSNheapNi':
                        # no inherit
                        alg = filter_search.EfficientBFSNoInherit(model)
                        alg.heap_class = 'simple'
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'EfficientBFSNheapN2':
                        # no ub2
                        alg = filter_search.EfficientBFS(model)
                        alg.heap_class = 'simple'
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic='ub0')
                    elif args.algorithm == 'BFSNheap':
                        # no efficient
                        alg = filter_search.InheritBFS(model)
                        alg.heap_class = 'simple'
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)
                    elif args.algorithm == 'AnytimeEfficientBFSNoInherit':
                        alg = filter_search.AnytimeEfficientBFSNoInherit(model)
                        alg.heap_class = 'simple'
                        alg.set_h(heuristic=ub)
                        alg.running_time = running_time
                        alg.report_interval = 10

                    elif args.algorithm == 'AnytimeEfficientBFS':
                        alg = filter_search.AnytimeEfficientBFS(model)
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic=ub)

                        alg.running_time = running_time
                        alg.report_interval = 10

                    elif args.algorithm == 'AnytimeEfficient':
                        alg = filter_search.AnytimeEfficientBranchAndBound(model)
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic='ub0')

                        alg.running_time = running_time
                        alg.report_interval = 10
                    elif args.algorithm == 'AnytimeBFSTC':
                        alg = filter_search.AnytimeBFSTC(model)
                        alg.use_alpha = False
                        alg.pushing_back = False
                        alg.set_d(d)
                        alg.set_h(heuristic='ub0')

                        alg.running_time = running_time
                        alg.report_interval = 10

                    alg.alpha = alpha
                    alg.setOpt(ub)
                    alg.build()
                    res = alg.optimize()
                    print(f"Done:seed:{seed}/{stop_seed - start_seed + 1}, ub:{ub}, d:{d}, budget:{budget}, res:{res}")

                    save_dir = os.path.join(root_dir, args.task, f'{args.num}', f'{seed}')
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)

                    save_path = os.path.join(save_dir, "{}-{}-{}-{}-{}-{}.pckl".format(
                        args.algorithm, ub, d, budget, running_time, model.__class__.__name__))

                    with open(save_path, "wb") as wrt:
                        pickle.dump(res, wrt)