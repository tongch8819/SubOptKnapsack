import json
import time

from budget_max_coverage import IdealMaxCovModel
from dblp_graph_coverage import DblpGraphCoverage
from facebook_graph_coverage import FacebookGraphCoverage
from image_sum import ImageSummarization
from movie_recommendation import MovieRecommendation
from revenue_max import RevenueMax, CalTechMaximization
from custom_coverage import CustomCoverage
from influence_maximization import YoutubeCoverage, CitationCoverage
from feature_selection import AdultIncomeFeatureSelection, SensorPlacement
from facility_location import MovieFacilityLocation

from greedy import greedy
from mgreedy import modified_greedy_ub1, modified_greedy_ub2, modified_greedy_ub3, modified_greedy_ub4,  modified_greedy_ub5, modified_greedy_ub1m, modified_greedy_ub7m, modified_greedy_ub7
from greedymax import greedy_max_ub1, greedy_max_ub2, greedy_max_ub3, greedy_max_ub4, greedy_max_ub4c, greedy_max_ub4cm, greedy_max_ub5, greedy_max_ub5c, greedy_max_ub6, greedy_max_ub5p, greedy_max_ub7, greedy_max_ub7m
from greedy_with_denstiy_threshold import gdt_ub1, gdt_ub2, gdt_ub3, gdt_ub4
from gcg import gcg_ub1, gcg_ub2, gcg_ub3, gcg_ub4

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import multiprocessing as mp
import argparse

cost_mode = "normal"
#upper_bounds = ["ub1", "ub3"]
upper_bounds = ["ub1m", "ub7m"]
# upper_bounds = ["ub7", "ub7m"]
# upper_bounds = ["ub1", "ub7", "ub7m"]
# algos = ["greedy_max", "modified_greedy"]
algos = ["modified_greedy"]
# algos = ["greedy_max"]
# algos = ["gcg"]
suffix = ""

# b graph ->
# replace with no vstar
# feature of minus sets
# submodular cost


knapsack = True
prepare_2_pair = False
print_curvature = True
graph_suffix = ""

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
                    algo, up + suffix, model.__class__.__name__, budget))
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
                    algo, up + suffix, model.__class__.__name__, budget))
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

    interval = 1
    num_points = 20
    start_point = 11
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    for budget in bds:
        model = MovieRecommendation(
            matrix_path="./dataset/movie/user_by_movies_small_rating.npy", budget=budget, k=30, n=500, knapsack=True, prepare_max_pair=False, print_curvature=False)
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)


def compute_dblp(root_dir, skip_mode=False):
    interval = 1
    num_points = 30
    start_point = 1
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    model = DblpGraphCoverage(
        budget=0, n=5000, graph_path="./dataset/com-dblp", knapsack=knapsack,
        prepare_max_pair=False, print_curvature=False, cost_mode=cost_mode, construct_graph=True)

    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
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
    start_point = 20
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    n = 1000
    s = f"-{n}"

    model = FacebookGraphCoverage(
        budget=0, n=n, seed=0, graph_path="./dataset/facebook", knapsack=knapsack, prepare_max_pair=False,
        print_curvature=False, cost_mode=cost_mode, construct_graph=True, graph_suffix=s)

    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_dir = os.path.join(root_dir, "archive-3", "facebook", f"{n}")
                save_path = os.path.join(save_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)

def compute_facebook_series(root_dir, skip_mode = False):
    n = 500
    seed_interval = 1
    start_seed = 0
    end_seed = 200

    for seed in range(start_seed, end_seed, seed_interval):
        start_time = time.time()

        interval = 1
        num_points = 25
        start_point = 6
        end_point = start_point + (num_points - 1) * interval
        bds = np.linspace(start=start_point, stop=end_point, num=num_points)
        s = f"-{n}"

        model = FacebookGraphCoverage(
            budget=0, n=n, seed=seed, graph_path="./dataset/facebook", knapsack=knapsack, prepare_max_pair=False,
            print_curvature=False, cost_mode=cost_mode, construct_graph=True, graph_suffix=s)

        save_dir = os.path.join(root_dir, "archive-5", "facebook", f"{n}", f"{seed}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for budget in bds:
            model.budget = budget
            for up in upper_bounds:
                for algo in algos:
                    save_path = os.path.join(save_dir, "{}-{}-{}-{:.2f}.pckl".format(
                        algo, up + suffix, model.__class__.__name__, budget))
                    func_call = eval(algo + "_" + up)
                    res = func_call(model)  # dict
                    if skip_mode and os.path.exists(save_path):
                        print("Skip: ", save_path)
                        continue
                    with open(save_path, "wb") as wrt:
                        pickle.dump(res, wrt)
                    print(res)
                    print("Done: ", save_path)

        stop_time = time.time()
        print(f"progress:{seed}/{end_seed} completed, total time:{stop_time-start_time}")

def compute_custom(root_dir, skip_mode=False):
    model = CustomCoverage(budget=10, n=100, graph_path="./dataset/custom-graph/graphs/100--0.2--0.1--0.5--20", knapsack=False, prepare_max_pair=prepare_2_pair,print_curvature=print_curvature)
    interval = 1
    num_points = 20
    start_point = 1
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
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

def compute_youtube(root_dir, skip_mode=False):
    n = 1000
    model = YoutubeCoverage(0, n, "./dataset/com-youtube", seed=1, knapsack=knapsack, cost_mode=cost_mode, prepare_max_pair=False,print_curvature=False, construct_graph=True)
    interval = 1
    num_points = 10
    start_point = 20
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(os.path.join(root_dir,"archive-3","youtube",f"{n}"), "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)

def compute_youtube_series(root_dir, skip_mode=False):
    n = 1000
    seed_interval = 1
    start_seed = 0
    end_seed = 200

    for seed in range(start_seed, end_seed, seed_interval):
        start_time = time.time()

        interval = 1
        num_points = 25
        start_point = 5
        end_point = start_point + (num_points - 1) * interval
        bds = np.linspace(start=start_point, stop=end_point, num=num_points)

        model = YoutubeCoverage(0, n, "./dataset/com-youtube", seed=seed, knapsack=knapsack, cost_mode=cost_mode,
                                prepare_max_pair=False, print_curvature=False, construct_graph=True)

        save_dir = os.path.join(root_dir, "archive-5", "youtube", f"{n}", f"{seed}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for budget in bds:
            model.budget = budget
            for up in upper_bounds:
                for algo in algos:
                    save_path = os.path.join(save_dir, "{}-{}-{}-{:.2f}.pckl".format(
                        algo, up + suffix, model.__class__.__name__, budget))
                    func_call = eval(algo + "_" + up)
                    res = func_call(model)  # dict
                    if skip_mode and os.path.exists(save_path):
                        print("Skip: ", save_path)
                        continue
                    with open(save_path, "wb") as wrt:
                        pickle.dump(res, wrt)
                    print(res)
                    print("Done: ", save_path)

        stop_time = time.time()
        print(f"progress:{seed}/{end_seed} completed, total time:{stop_time-start_time}")

def compute_citation(root_dir, skip_mode=False):
    model = CitationCoverage(0, 1000, "./dataset/cite-HepPh", knapsack=knapsack, prepare_max_pair=False, cost_mode=cost_mode)
    interval = 1
    num_points = 30
    start_point = 1
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)

def compute_caltech(root_dir, skip_mode=False):
    n = 100
    s = f"-{n}"
    model = CalTechMaximization(0, n,"./dataset/caltech",seed=21, knapsack=True, prepare_max_pair=False, cost_mode=cost_mode,print_curvature=False, graph_suffix=s, construct_graph=True)
    interval = 1
    num_points = 1
    start_point = 15
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(os.path.join(root_dir, "archive-3", "caltech", f"{n}"), "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)

def compute_caltech_series(root_dir, skip_mode = False):
    n = 50
    seed_interval = 1
    start_seed = 0
    end_seed = 200

    for seed in range(start_seed, end_seed, seed_interval):
        start_time = time.time()

        interval = 1
        num_points = 25
        start_point = 6
        end_point = start_point + (num_points - 1) * interval
        bds = np.linspace(start=start_point, stop=end_point, num=num_points)
        s = f"-{n}"

        model = CalTechMaximization(0, n, "./dataset/caltech", seed=seed, knapsack=True, prepare_max_pair=False,
                                    cost_mode=cost_mode, print_curvature=False, graph_suffix=s, construct_graph=True)

        save_dir = os.path.join(root_dir, "archive-5", "caltech", f"{n}", f"{seed}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for budget in bds:
            model.budget = budget
            for up in upper_bounds:
                for algo in algos:
                    save_path = os.path.join(save_dir, "{}-{}-{}-{:.2f}.pckl".format(
                                                 algo, up + suffix, model.__class__.__name__, budget))
                    func_call = eval(algo + "_" + up)
                    res = func_call(model)  # dict
                    if skip_mode and os.path.exists(save_path):
                        print("Skip: ", save_path)
                        continue
                    with open(save_path, "wb") as wrt:
                        pickle.dump(res, wrt)
                    print(res)
                    print("Done: ", save_path)

        stop_time = time.time()
        print(f"progress:{seed}/{end_seed} completed, total time:{stop_time-start_time}")

def compute_adult(root_dir, skip_mode=False):
    n = 100
    sample_count = 100
    model = AdultIncomeFeatureSelection(0, n, "./dataset/adult-income", sample_count=sample_count, knapsack=True, construct_graph=True)
    interval = 1
    num_points = 20
    start_point = 1
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(os.path.join(root_dir, "archive-3", "adult", f"{sample_count}", f"{n}"), "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)

def compute_adult_series(root_dir, skip_mode = False):
    n = 50
    sample_count = 100

    seed_interval = 1
    start_seed = 0
    end_seed = 200

    for seed in range(start_seed, end_seed, seed_interval):
        start_time = time.time()

        interval = 1
        num_points = 25
        start_point = 6
        end_point = start_point + (num_points - 1) * interval
        bds = np.linspace(start=start_point, stop=end_point, num=num_points)

        model = AdultIncomeFeatureSelection(0, n, "./dataset/adult-income", seed=seed, sample_count=100, knapsack=True, construct_graph=True)

        save_dir = os.path.join(root_dir, "archive-5", "adult", f"{n}", f"{seed}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for budget in bds:
            model.budget = budget
            for up in upper_bounds:
                for algo in algos:
                    save_path = os.path.join(save_dir, "{}-{}-{}-{:.2f}.pckl".format(
                                                 algo, up + suffix, model.__class__.__name__, budget))
                    func_call = eval(algo + "_" + up)
                    res = func_call(model)  # dict
                    if skip_mode and os.path.exists(save_path):
                        print("Skip: ", save_path)
                        continue
                    with open(save_path, "wb") as wrt:
                        pickle.dump(res, wrt)
                    print(res)
                    print("Done: ", save_path)

        stop_time = time.time()
        print(f"progress:{seed}/{end_seed} completed, total time:{stop_time-start_time}")

def compute_sensor(root_dir, skip_mode=False):
    model = SensorPlacement(0, 100, "./dataset/berkley-sensor", knapsack=True, construct_graph=False,cost_mode=cost_mode)
    interval = 1
    num_points = 23
    start_point = 1
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)
    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)

def compute_facility(root_dir, skip_mode=False):
    # budget = 30.0
    # model = MovieRecommendation(matrix_path="/home/ctong/Projects/SubOptKnapsack/dataset/movie/user_by_movies_small_rating.npy", budget=budget, k = 30, n = 50)
    # print(model.num_movies, model.num_users)
    # res = greedy_max_ub1(model)
    # print(res)

    interval = 1
    num_points = 30
    start_point = 11
    end_point = start_point + (num_points - 1) * interval
    bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    model = MovieFacilityLocation(
        matrix_path="./dataset/movie/user_by_movies_small_rating.npy", budget=0, k=30, n=1000, knapsack=True,
        prepare_max_pair=False, print_curvature=False)

    for budget in bds:
        model.budget = budget
        for up in upper_bounds:
            for algo in algos:
                save_path = os.path.join(root_dir, "{}-{}-{}-{:.2f}.pckl".format(
                    algo, up + suffix, model.__class__.__name__, budget))
                func_call = eval(algo + "_" + up)
                res = func_call(model)  # dict
                if skip_mode and os.path.exists(save_path):
                    print("Skip: ", save_path)
                    continue
                with open(save_path, "wb") as wrt:
                    pickle.dump(res, wrt)
                print(res)
                print("Done: ", save_path)

def run_multi_custom_exps(config_path: str):
    # run custom_coverage with different n, alpha, beta, gamma
    # compute the c2, d, e and curvature for each model
    # compute the mean difference between the AF of ub3 and ub5, MDAF
    # draw a plot showing the relationship between n, c2, d, e, curvature, alpha, beta, gamma and MDAF

    config_s = ""

    with open(config_path, "r") as f:
        config_list = f.readlines()
        for c in config_list:
            config_s = config_s + c
        config_s.replace("\n", "")

    coverage_dir = "archive/custom-coverage-ub77m"

    graphs_path = "dataset/custom-graph/graphs"

    graphs = os.listdir(graphs_path)

    for graph_path in graphs:
        model = CustomCoverage(budget=10, graph_path=os.path.join(graphs_path, graph_path),
                               knapsack=False, prepare_max_pair=prepare_2_pair,
                               print_curvature=print_curvature)
        interval = 1
        num_points = 5
        start_point = 5
        end_point = start_point + (num_points - 1) * interval
        bds = np.linspace(start=start_point, stop=end_point, num=num_points)

        algo = "greedy_max"

        AF_3 = 0.
        AF_3_time = 0.

        AF_5 = 0.
        AF_5_time = 0.

        for budget in bds:
            model.budget = budget

            start_time = time.time()

            func_call = eval(algo + "_ub7")
            res = func_call(model)  # dict
            AF_3 += res["AF"]

            AF_3_time += time.time() - start_time
            start_time = time.time()

            func_call = eval(algo + "_ub7m")
            res = func_call(model)  # dict
            AF_5 += res["AF"]

            AF_5_time += time.time() - start_time

        MDAF = (AF_5 * 100 - AF_3*100)/num_points

        n, alpha, beta, gamma, seed = graph_path.split("--")

        c2 = model.c2
        d = model.d
        e = model.e
        c = model.curvature

        res = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "n": n,

            "curvature": c,
            "c2": c2,
            "d": d,
            "e": e,

            "MDAF": MDAF,

            "3Time": AF_3_time,
            "5Time": AF_5_time
        }

        print(f"AF_3_time:{AF_3_time}, AF_5_time:{AF_5_time}")

        save_path = os.path.join(root_dir, coverage_dir, "{}-{}-{}-{}-{}-{}.pckl".format(
            model.__class__.__name__, alpha, beta, gamma, n, seed))

        with open(save_path, "wb") as wrt:
            pickle.dump(res, wrt)

        print(res)
        print(f"Done: alpha:{alpha}, beta:{beta}, gamma:{gamma}, seed:{seed}, MDAF:{MDAF}")

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

    config_path = "./dataset/custom-graph/config.json"

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
    elif args.task_num == 6:
        compute_custom(root_dir)
    elif args.task_num == 7:
        run_multi_custom_exps(config_path)
    elif args.task_num == 8:
        compute_youtube(root_dir)
    elif args.task_num == 9:
        compute_citation(root_dir)
    elif args.task_num == 10:
        compute_caltech(root_dir)
    elif args.task_num == 11:
        compute_adult(root_dir)
    elif args.task_num == 12:
        compute_sensor(root_dir)
    elif args.task_num == 13:
        compute_facility(root_dir)
    elif args.task_num == 14:
        compute_facebook_series(root_dir)
    elif args.task_num == 15:
        compute_caltech_series(root_dir)
    elif args.task_num == 16:
        compute_youtube_series(root_dir)
    elif args.task_num == 17:
        compute_adult_series(root_dir)