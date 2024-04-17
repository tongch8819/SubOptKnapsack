from budget_max_coverage import IdealMaxCovModel
from dblp_graph_coverage import DblpGraphCoverage
from facebook_graph_coverage import FacebookGraphCoverage
from image_sum import ImageSummarization
from movie_recommendation import MovieRecommendation
from revenue_max import RevenueMax

from sieve_streaming import sieve_streaming_ub0, \
    sieve_streaming_ub1, \
    sieve_streaming_ub2, sieve_streaming_ub3, sieve_streaming_ub4, sieve_one_pass_streaming_ub0, \
    sieve_one_pass_streaming_ub1 \

from sieve_streaming import sieve_knapsack_streaming_ub0, sieve_streaming_ub1, sieve_knapsack_streaming_ub1


import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import multiprocessing as mp
import argparse

# upper_bounds = ["ub1", "ub3"]
upper_bounds = ["ub0", "ub1"]

# upper_bounds = ["ub4"]
# algos = ["greedy_max", "modified_greedy"]
# algos = ["greedy_max"]

algos = ["sieve_streaming"]
# algos = ["sieve_knapsack_streaming"]

knapsack = False

def compute_max_cov(root_dir, skip_mode=False):
    interval = 0.3
    num_points = 20
    start_point = 0.5
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    for budget in bds:
        model = IdealMaxCovModel(budget=budget)
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up, model.__class__.__name__, budget))
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print("Done: ", save_path)


def compute_dblp(root_dir, skip_mode=False):
    interval = 1
    num_points = 10
    start_point = 5
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model = DblpGraphCoverage(
            budget=budget, n=5000, graph_path="./dataset/com-dblp/com-dblp.top5000.cmty.txt", knapsack=knapsack)
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, 'ub5', model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)


def compute_facebook(root_dir, skip_mode=False):
    interval = 1
    num_points = 10
    start_point = 5
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model = FacebookGraphCoverage(
            budget=budget, n=5000, graph_path="./dataset/facebook/facebook_combined.txt", knapsack=knapsack)
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print("Done: ", save_path)


def run_multiple_exps(root_dir, skip_mode):
    result_lst = []
    sufs = ["max_cov", "image_sum", "movie_recom", "revenue_max"]
    with mp.Pool() as pool:
        for suffix in sufs:
            func_call = eval("compute_{}".format(suffix))  # eval string into function object
            result = pool.apply_async(func_call, [root_dir, skip_mode])
            result_lst.append(result)
        [res.wait() for res in result_lst]


if __name__ == "__main__":
    root_dir = "./result"

    parser = argparse.ArgumentParser()
    parser.add_argument("task_num", type=int, help="0,1,2,3,4")
    args = parser.parse_args()

    if args.task_num == 0:
        compute_max_cov(root_dir)
    elif args.task_num == 4:
        compute_dblp(root_dir)
    elif args.task_num == 5:
        compute_facebook(root_dir)

    # run_multiple_exps(root_dir, True)