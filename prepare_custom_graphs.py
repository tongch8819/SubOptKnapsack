import json
import math
import random

from budget_max_coverage import IdealMaxCovModel
from dblp_graph_coverage import DblpGraphCoverage
from facebook_graph_coverage import FacebookGraphCoverage
from image_sum import ImageSummarization
from movie_recommendation import MovieRecommendation
from revenue_max import RevenueMax, CalTechMaximization
from custom_coverage import CustomCoverage
from influence_maximization import YoutubeCoverage, CitationCoverage

from greedy import greedy
from mgreedy import modified_greedy_ub1, modified_greedy_ub2, modified_greedy_ub3, modified_greedy_ub4,  modified_greedy_ub5
from greedymax import greedy_max_ub1, greedy_max_ub2, greedy_max_ub3, greedy_max_ub4, greedy_max_ub4c, greedy_max_ub5, greedy_max_ub5c, greedy_max_ub6
from greedy_with_denstiy_threshold import gdt_ub1, gdt_ub2, gdt_ub3, gdt_ub4
from gcg import gcg_ub1, gcg_ub2, gcg_ub3, gcg_ub4

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import multiprocessing as mp
import argparse
import networkx as nx
knapsack = False
prepare_2_pair = False
print_curvature = True
max_nodes = 100
graph_path = "dataset/custom-graph/custom.txt"

import pandas as pd

def prepare_graphs(config_path: str):
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

    config = json.loads(config_s)

    n = config["n"]
    alpha_min = config["alpha"]["min"]
    alpha_max = config["alpha"]["max"]
    alpha_step = config["alpha"]["step"]

    beta_min = config["beta"]["min"]
    beta_max = config["beta"]["max"]
    beta_step = config["beta"]["step"]

    gamma_min = config["gamma"]["min"]
    gamma_max = config["gamma"]["max"]
    gamma_step = config["gamma"]["step"]

    seed_min = config["seed"]["min"]
    seed_max = config["seed"]["max"]
    seed_step = config["seed"]["step"]

    alpha = alpha_min

    graph_dir = "dataset/custom-graph/kgraphs"

    while alpha <= alpha_max:
        beta = beta_min
        while beta <= beta_max:
            gamma = gamma_min
            while gamma <= gamma_max:
                for seed in range(seed_min, seed_max, seed_step):
                    model_name = f"{n}--{alpha}--{beta}--{gamma}--{seed}"
                    model = CustomCoverage(budget=10, n=n, alpha=alpha, beta=beta, seed=seed, gamma=gamma,
                                           graph_path=graph_dir+"/"+model_name,
                                           knapsack=knapsack, prepare_max_pair=prepare_2_pair,
                                           print_curvature=print_curvature,construct_graph=True)
                gamma += gamma_step

            beta += beta_step

        alpha += alpha_step

def load_graph(path: str):
    if not os.path.isfile(path):
        raise OSError("File *.txt does not exist.")
    intact_graph: nx.Graph = nx.read_adjlist(path)
    nodes = random.sample(list(intact_graph.nodes), max_nodes)

    return intact_graph.subgraph(nodes)

def prepare_feature_selection():
    data_path = "./dataset/adult-income/adult.csv"

    df = pd.read_csv(data_path, encoding="utf-8")

    data = np.array(df)

    y = data[:, 14]

    y = [
        [1 if i == '>50K' else 0]
        for i in y
    ]

    y = np.array(y)

    x_s = np.ndarray(shape=(data.shape[0], 0))
    feature_count = 0
    # 将x拆分成binary feature
    for i in range(0, 14):
        x_i = data[:, i]
        all_digit = True
        printed = False
        for f in x_i:
            if not printed:
                printed = True
            if type(f) != int and f != '?':
                all_digit = False
                break
        if all_digit:
            max_x = max(x_i)
            count = int(math.ceil(math.log(max_x, 2))) + 1
            feature_count += count

            x_i_binary = []
            for s in x_i:
                o_bin = bin(s)[2:]
                o_bin = o_bin.rjust(count, '0')
                o_bin = list(o_bin)
                o_bin = [int(b) for b in o_bin]
                x_i_binary.append(o_bin)

        else:
            t_x_i = set(x_i)
            values = len(t_x_i)
            count = int(math.ceil(math.log(values, 2))) + 1
            feature_count += count
            t_x_i = list(t_x_i)
            binary_dict = {}
            for idx in range(0, len(t_x_i)):
                o_bin = bin(idx)[2:]
                o_bin = o_bin.rjust(count, '0')
                o_bin = list(o_bin)
                o_bin = [int(b) for b in o_bin]
                binary_dict[t_x_i[idx]] = o_bin

            x_i_binary = []
            for s in x_i:
                x_i_binary.append(binary_dict[s])

        x_i_binary = np.array(x_i_binary)
        x_s = np.append(x_s, x_i_binary, axis = 1)

    print(x_s.shape)
    print(y.shape)

    df = np.append(x_s, y, axis= 1)
    print(df.shape)

    with open("./dataset/adult-income/binary_data.txt", 'w') as f:
        for i in range(0, df.shape[0]):
            s = df[i]
            s_str = ""
            for feat in s:
                s_str = s_str + f"{feat} "
            s_str += "\n"
            f.write(s_str)

    pass

# 读取每个sensor的前200行数据
def prepare_sensor_placement(n = 1000):
    current_sensor_idx = 1
    current_sensor_data = n

    temps = [[]]

    with open("./dataset/berkley-sensor/data.txt", "r") as f:
        while True:
            line = f.readline()
            if line == "":
                # print(line)
                # print(current_sensor_idx)
                # print(current_sensor_data)
                break
            else:
                date, time, epoch, idx, temp, humid, light, voltage = list(line.rstrip("\n").split(" "))
                if idx.isdigit() and not temp == '':
                    idx = int(idx)
                    if idx == current_sensor_idx:
                        current_sensor_data -= 1
                        temps[idx-1].append(float(temp))

                        if current_sensor_data == 0:
                            current_sensor_idx += 1
                            if current_sensor_idx > 51:
                                # print("1")
                                # print(line)
                                # print(current_sensor_idx)
                                # print(current_sensor_data)
                                break
                            current_sensor_data = n
                            temps.append([])

                            while True:
                                line = f.readline()
                                if line == "":
                                    break
                                else:
                                    date, time, epoch, idx, temp, humid, light, voltage = list(line.rstrip("\n").split(" "))
                                    if idx.isdigit():
                                        idx = int(idx)
                                        if idx == current_sensor_idx:
                                            if not temp == '':
                                                current_sensor_data -= 1
                                                temps[idx - 1].append(float(temp))
                                            break

                    else:
                        current_sensor_idx += 1
                        if current_sensor_idx > 50:
                            break
                        current_sensor_data = n

                        temps.append([])


    # 5号传感器异常
    # print(len(temps))
    temps.remove([])
    for i in range(0, len(temps)):
        print(f"i:{i}, f:{len(temps[i])}")

    with open("./dataset/berkley-sensor/t_data.txt", "w") as f:
        for i in range(0, len(temps)):
            t = ""
            for j in range(0, len(temps[i])):
                if j == len(temps[i]) - 1:
                    t = t + f"{temps[i][j]}"
                else:
                    t = t + f"{temps[i][j]} "
            t = t + "\n"
            f.write(t)

def prepare_facebook():
    facebook = FacebookGraphCoverage(0, 1000, graph_path="./dataset/facebook/graphs/1", knapsack=True,
                                     prepare_max_pair=False, print_curvature=False, construct_graph=True,
                                     graph_suffix="-1000")

    g1 = facebook.graph
    c1 = facebook.costs_obj
    g1l = list(g1.nodes)
    g1l.sort()
    print(f"g1:{g1l[:10]}")

    facebook = FacebookGraphCoverage(0, 1000, graph_path="./dataset/facebook/graphs/1", knapsack=True,
                                     prepare_max_pair=False, print_curvature=False, construct_graph=True,
                                     graph_suffix="-1000")

    g2 = facebook.graph
    c2 = facebook.costs_obj
    g2l = list(g2.nodes)
    g2l.sort()
    print(f"g2:{g2l[:10]}")

def prepare_caltech():
    cal = CalTechMaximization(0, 100, "./dataset/caltech", knapsack=True, prepare_max_pair=False, construct_graph=True,
                              graph_suffix="-100100")



if __name__ == "__main__":
    # prepare_facebook()

    # run_multiple_exps(root_dir, True)

    prepare_caltech()