import matplotlib.pyplot as plt
import os.path
import time

import numpy as np

from facebook_graph_coverage import FacebookGraphCoverage
from feature_selection import AdultIncomeFeatureSelection
from influence_maximization import YoutubeCoverage
from revenue_max import CalTechMaximization

cost_mode = "normal"

def parse_m():
    result_dir = os.path.join("./result", "da")

    tasks_list = ["facebook", "youtube", "caltech", "adult"]

    t_mm_list = []

    start = 0
    stop = 1
    count = stop - start

    for task in tasks_list:
        mm_list = []

        for seed in range(start, stop):
            start_time = time.time()

            model = None
            if task == "facebook":
                model = FacebookGraphCoverage(
                    budget=0, n=1000, seed=seed, graph_path="./dataset/facebook", knapsack=True, prepare_max_pair=False,
                    print_curvature=False, cost_mode=cost_mode, construct_graph=True, graph_suffix="")
            elif task == "youtube":
                model = YoutubeCoverage(0, 1000, "./dataset/com-youtube", seed=seed, knapsack=True, cost_mode=cost_mode,
                                prepare_max_pair=False, print_curvature=False, construct_graph=True)
            elif task == "caltech":
                model = CalTechMaximization(0, 100, "./dataset/caltech", seed=seed, knapsack=True, prepare_max_pair=False,
                                            cost_mode=cost_mode, print_curvature=False, graph_suffix="",
                                            construct_graph=True)
            elif task == "adult":
                model = AdultIncomeFeatureSelection(0, 100, "./dataset/adult-income", seed=seed, sample_count=100,
                                                    knapsack=True, construct_graph=True)

            m = model.calculate_m()
            mm_list.append(np.mean(m))

            stop_time = time.time()

            print(f"{task}:{seed+1}/{count}, computational time:{stop_time-start_time}")

        t_mm_list.append(mm_list)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_ylabel("MM")

    box_props = dict(linewidth=1.5, color="black")
    median_props = dict(linewidth=2)
    mean_props = {"marker": "^", "markerfacecolor": "darkgreen", "markeredgecolor": "darkgreen"}
    flier_props = {"marker": "o", "markerfacecolor": "darkgreen", "markeredgecolor": "black"}
    whisker_props = {"color": "black"}

    bp = ax.boxplot(t_mm_list, patch_artist=True, tick_labels=tasks_list,
                    showmeans=True, boxprops=box_props, medianprops=median_props
                    , meanprops=mean_props, flierprops=flier_props, whiskerprops=whisker_props, capprops=whisker_props)
    for median in bp["medians"]:
        median.set_color("black")
    for box in bp["boxes"]:
        box.set_facecolor("whitesmoke")

    plt.savefig(os.path.join(result_dir, f"tmm_{count}.png"), bbox_inches="tight")

    pass

def parse_m_2():
    result_dir = os.path.join("./result", "da")

    tasks_list = ["facebook", "youtube", "caltech", "adult"]

    label_list = {"facebook" : "ego-facebook",
                  "youtube": "com-youtube",
                  "caltech" : "Caltech36",
                  "adult" : "Adult Income"}

    t_mm_list = []

    fs = 24

    font = {'family': 'normal',
            'size': fs}

    plt.rc('font', **font)

    start = 0
    stop = 200
    count = stop - start

    for task in tasks_list:
        mm_list = []

        for seed in range(start, stop):
            start_time = time.time()

            model = None
            if task == "facebook":
                model = FacebookGraphCoverage(
                    budget=0, n=1000, seed=seed, graph_path="./dataset/facebook", knapsack=True, prepare_max_pair=False,
                    print_curvature=False, cost_mode=cost_mode, construct_graph=True, graph_suffix="")
            elif task == "youtube":
                model = YoutubeCoverage(0, 1000, "./dataset/com-youtube", seed=seed, knapsack=True, cost_mode=cost_mode,
                                prepare_max_pair=False, print_curvature=False, construct_graph=True)
            elif task == "caltech":
                model = CalTechMaximization(0, 100, "./dataset/caltech", seed=seed, knapsack=True, prepare_max_pair=False,
                                            cost_mode=cost_mode, print_curvature=False, graph_suffix="",
                                            construct_graph=True)
            elif task == "adult":
                model = AdultIncomeFeatureSelection(0, 100, "./dataset/adult-income", seed=seed, sample_count=100,
                                                    knapsack=True, construct_graph=True)

            m = model.calculate_m()
            mm_list = mm_list + m

            stop_time = time.time()

            print(f"{task}:{seed+1}/{count}, computational time:{stop_time-start_time}")

        t_mm_list.append(mm_list)

    with open("kdkdk.txt", "w") as f:
        for l in t_mm_list:
            f.write(f"{l} ")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_ylabel(r"$d_{V \setminus \{e\}}(e)$")

    box_props = dict(linewidth=1.5, color="black")
    median_props = dict(linewidth=2)
    mean_props = {"marker": "^", "markerfacecolor": "darkgreen", "markeredgecolor": "darkgreen"}
    flier_props = {"marker": "o", "markerfacecolor": "darkgreen", "markeredgecolor": "black"}
    whisker_props = {"color": "black"}


    bp = ax.boxplot(t_mm_list, patch_artist=True,
                    showmeans=True, boxprops=box_props, medianprops=median_props
                    , meanprops=mean_props, flierprops=flier_props, whiskerprops=whisker_props, capprops=whisker_props)

    ax.set_xticks([i+1 for i in range(0, len(tasks_list))])
    ax.set_xticklabels(label_list[task] for task in tasks_list)

    for median in bp["medians"]:
        median.set_color("black")
    for box in bp["boxes"]:
        box.set_facecolor("whitesmoke")

    plt.savefig(os.path.join(result_dir, f"tmm2_{count}.pdf"), bbox_inches="tight")
    plt.clf()

    pass

def open_analyzer():
    st_mm_list = []
    with open("kdkdk.txt", "r") as f:
        st_mm_list.append(f.readline())

    print(len(st_mm_list))

    t_mm_list = []
    for l in st_mm_list:
        l = l.lstrip('[').rstrip(']')
        print(l)

if __name__ == "__main__":
    parse_m_2()