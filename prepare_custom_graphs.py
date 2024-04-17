import json
import random

from budget_max_coverage import IdealMaxCovModel
from dblp_graph_coverage import DblpGraphCoverage
from facebook_graph_coverage import FacebookGraphCoverage
from image_sum import ImageSummarization
from movie_recommendation import MovieRecommendation
from revenue_max import RevenueMax
from custom_coverage import CustomCoverage

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

if __name__ == "__main__":

    prepare_graphs("dataset/custom-graph/config.json")


    # run_multiple_exps(root_dir, True)