from budget_max_coverage import IdealMaxCovModel
from dblp_graph_coverage import DblpGraphCoverage
from facebook_graph_coverage import FacebookGraphCoverage
from image_sum import ImageSummarization
from movie_recommendation import MovieRecommendation
from revenue_max import RevenueMax

from greedy import greedy
from mgreedy import modified_greedy_ub1, modified_greedy_ub2, modified_greedy_ub3, modified_greedy_ub4
from greedymax import greedy_max_ub1, greedy_max_ub2, greedy_max_ub3, greedy_max_ub4

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import multiprocessing as mp
import argparse


#upper_bounds = ["ub1", "ub3"]
upper_bounds = ["ub3", "ub4"]
# upper_bounds = ["ub4"]
# algos = ["greedy_max", "modified_greedy"]
algos = ["greedy_max"]

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


def compute_image_sum(root_dir, skip_mode=False):
    # costs = model.costs_obj
    # max(costs), min(costs)
    # 8.302965859527456 2.3885016618205253
    interval = 2
    num_points = 20
    start_point = 5
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    max_num = 50

    for budget in bds:
        model = ImageSummarization(
            image_path="/home/ctong/Projects/SubOptKnapsack/dataset/image/500_cifar10_sample.npy", budget=budget, max_num=max_num)
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

    # budget = 10.0
    # model = ImageSummarization(
    #     image_path="/home/ctong/Projects/SubOptKnapsack/dataset/image/500_cifar10_sample.npy", budget=budget)
    # res = greedy_max_ub1(model)
    # print(res)


def compute_movie_recom(root_dir, skip_mode=False):
    # budget = 30.0
    # model = MovieRecommendation(matrix_path="/home/ctong/Projects/SubOptKnapsack/dataset/movie/user_by_movies_small_rating.npy", budget=budget, k = 30, n = 50)
    # print(model.num_movies, model.num_users)
    # res = greedy_max_ub1(model)
    # print(res)

    interval = 2
    num_points = 20
    start_point = 5
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    for budget in bds:
        model = MovieRecommendation(
            matrix_path="./dataset/movie/user_by_movies_small_rating.npy", budget=budget, k=30, n=50)
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

def compute_dblp(root_dir, skip_mode=False):
    interval = 1
    num_points = 20
    start_point = 1
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model = DblpGraphCoverage(
            budget=budget, n=5000, graph_path="./dataset/com-dblp/com-dblp.top5000.cmty.txt")
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
                print(res)
                print("Done: ", save_path)

def compute_facebook(root_dir, skip_mode=False):
    interval = 1
    num_points = 20
    start_point = 1
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model = FacebookGraphCoverage(
            budget=budget, n=5000, graph_path="./dataset/facebook/facebook_combined.txt")
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

def compute_revenue_max(root_dir, skip_mode=False):
    # budget = 1
    # model = RevenueMax(budget=budget, pckl_path="/home/ctong/Projects/SubOptKnapsack/dataset/revenue/25_youtube_top5000.pkl")
    # costs = model.costs_obj
    # print(max(costs), min(costs))
    # # 0.6138416218800486 0.018249099326685947

    interval = 0.1
    num_points = 20
    start_point = 0.05
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)


    model = RevenueMax(budget=1.0, pckl_path="/home/ctong/Projects/SubOptKnapsack/dataset/revenue/25_youtube_top5000.pkl")

    for budget in bds:
        model.b = budget
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
    elif args.task_num == 1:
        compute_image_sum(root_dir)
    elif args.task_num == 2:
        compute_movie_recom(root_dir)
    elif args.task_num == 3:
        compute_revenue_max(root_dir)
    elif args.task_num == 4:
        compute_dblp(root_dir)
    elif args.task_num == 5:
        compute_facebook(root_dir)

    # run_multiple_exps(root_dir, True)