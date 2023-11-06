from budget_max_coverage import IdealMaxCovModel
from greedy import greedy
from mgreedy import modified_greedy_ub1, modified_greedy_ub2, modified_greedy_ub3
from greedymax import greedy_max_ub1, greedy_max_ub2, greedy_max_ub3

import matplotlib.pyplot as plt
import numpy as np
import pickle

# for budget in 
interval = 0.3
num_points = 20
start_point = 0.5
end_point = start_point + (num_points - 1) * interval
bds = np.linspace(start=start_point, stop=end_point, num=num_points)
upper_bounds = ["ub1", "ub3", "ub2"]
# upper_bounds = ["ub2"]
# algos = ["modified_greedy", "greedy_max"]
algos = ["greedy_max"]

for budget in bds:
    model = IdealMaxCovModel(b=budget)
    for up in upper_bounds:
        for algo in algos:
            func_call = eval(algo + "_" + up)
            res = func_call(model)  # dict
            save_path = "./result/{}-{}-{}.pckl".format(algo, up, budget)
            with open(save_path, "wb") as wrt:
                pickle.dump(res, wrt)
            print("Done: ", save_path)