from application.movie_recommendation_knapsack import MovieRecommendationKnapsack
from application.model import model_constructor

from algorithms.greedy import greedy_density_knapsack, greedy_marginal_gain_knapsack
from algorithms.greedy import is_dominance_theoretical

import numpy as np
import os
from typing import List
import logging
import argparse
from tqdm import tqdm
logging.basicConfig(level=logging.ERROR)





def is_dominance_practical(density_res, marginal_gain_res):
    return density_res['f(S)'] >= marginal_gain_res['f(S)']


def unit_test_single_run():
    budget_ratio = 0.1
    model = MovieRecommendationKnapsack(n=50, k=50, budget_ratio=budget_ratio)
    model.constraint.normalize()

    cost_ratio = max(model.constraint.cost_func)
    budget = model.constraint.budget

    print(f"budget: {budget}, cost_ratio: {cost_ratio}, budget_ratio: {model.constraint.budget / sum(model.constraint.cost_func)}")

    density_res = greedy_density_knapsack(model)
    print("Density greedy: ", density_res)
    marginal_gain_res = greedy_marginal_gain_knapsack(model)
    print("Marginal gain greedy: ", marginal_gain_res)

    print("Is density dominant? (Theorectical)",
          is_dominance_theoretical(cost_ratio, budget))
    print("Is density dominant? (Practical)",
          is_dominance_practical(density_res, marginal_gain_res))


def search_rb_points(max_num_points=10, output_file="dominance.txt"):
    dominance = []
    not_dominance = []
    for cost_ratio in np.arange(1.1, 5.1, 0.2):
        for budget in np.arange(cost_ratio + 0.1, 5.1, 0.2):
            if is_dominance_theoretical(cost_ratio, budget):
                # logging.debug(f"dominance: cost_ratio: {cost_ratio}, budget: {budget}")
                if len(dominance) < max_num_points:
                    dominance.append((cost_ratio, budget))
            else:
                # logging.debug(f"NOT dominance: cost_ratio: {cost_ratio}, budget: {budget}")
                if len(not_dominance) < max_num_points:
                    not_dominance.append((cost_ratio, budget))

    if len(dominance) < max_num_points:
        logging.warning(f"dominance length is not enough: {len(dominance)}")
    if len(not_dominance) < max_num_points:
        logging.warning(
            f"not_dominance length is not enough: {len(not_dominance)}")
    # write to file
    with open(output_file, "w") as f:
        f.write("cost_ratio,budget,is_dominance\n")
        for item in dominance:
            f.write(f"{item[0]},{item[1]},True\n")
        for item in not_dominance:
            f.write(f"{item[0]},{item[1]},False\n")



def run_single_experiment(cost_ratio: float, budget: float, is_dominance_theoretical: bool, model_name: str):
    model = model_constructor(model_name)
    model.constraint.enforce_cost_ratio_and_budget(cost_ratio, budget)
    # logging.debug(model.constraint.show_stat())
    density_res = greedy_density_knapsack(model)
    marginal_gain_res = greedy_marginal_gain_knapsack(model)
    logging.debug(f"Density result: S={density_res['S']}, f(S)={density_res['f(S)']}; Marginal gain result: S={marginal_gain_res['S']}, f(S)={marginal_gain_res['f(S)']}")
    return (model_name, cost_ratio, budget, density_res['f(S)'], marginal_gain_res['f(S)'], is_dominance_theoretical, density_res['f(S)'] >= marginal_gain_res['f(S)'])


def run_batch(rb_pair_file_path: str, model_name_lst: List[str] = ["MovieRecommendation", "ImageSum"], output_file: str = "output.txt"):
    # load the data from the specified file
    with open(rb_pair_file_path, "r") as f:
        data = f.readlines()[1:]
        data = [x.strip().split(",") for x in data]
        data = [(float(x[0]), float(x[1]), eval(x[2])) for x in data]
    # with data, it contains lines of cost_ratio, budget, is_dominance

    res = []
    for model_name in model_name_lst:
        for cost_ratio, budget, is_dominance in tqdm(data, desc=f"Processing {model_name} models"):
            res_line = run_single_experiment(cost_ratio, budget, is_dominance, model_name)
            logging.debug(res_line)
            res.append(",".join(map(str, res_line)))

    with open(output_file, "w") as f:
        f.write("model_name,cost_ratio,budget,density,marginal_gain,is_dominance_theoretical,is_dominance_practical\n")
        for item in res:
            f.write(f"{item}\n")




def main():
    dump_dir = "result/CostInterpolation-20250207/test"
    parser = argparse.ArgumentParser(description="Run cost ratio interpolation experiments.")
    parser.add_argument("--maximum_points", type=int, default=5, help="Maximum number of points for dominance and non-dominance.")
    args = parser.parse_args()

    maximum_points = args.maximum_points
    sample_pair_path = os.path.join(dump_dir, f"rb_sample_pair_{maximum_points}.txt")
    output_file = os.path.join(dump_dir, f"output_{maximum_points}.txt")

    # generate two groups of dominance and non-dominance
    search_rb_points(max_num_points=maximum_points, output_file=sample_pair_path)
    # these points could be used for all applications
    run_batch(sample_pair_path, model_name_lst=["MovieRecommendation", "ImageSum", "MaxFLP", "MaxRevenue"], output_file=output_file)
    # run_batch(sample_pair_path, model_name_lst=["MaxRevenue"], output_file=output_file)
    


if __name__ == "__main__":
    main()
