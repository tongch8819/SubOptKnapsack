from application.model import model_constructor

from algorithms.greedy import greedy_density_knapsack, greedy_marginal_gain_knapsack, greedy_max, modified_greedy, adaptive_greedy, greedy_aggregate

import numpy as np
from typing import List
import argparse
import os
import logging
from tqdm import tqdm


# aggregate must be located after density and marginal gain
baselines = {
    "Density" : greedy_density_knapsack,
    "Marginal gain" : greedy_marginal_gain_knapsack,
    "Aggregate" : greedy_aggregate,
    "MGreedy" : modified_greedy,
    "GreedyMax" : greedy_max,
    "AdaptiveGreedy" : adaptive_greedy,
}


def generate_rb_grid(maximum_points=5): 
    """
    Generate grid for cost ratio and budget
    Warning: does not check if cost_ratio >= budget
    """
    cost_ratio_seq = np.linspace(1.1, 5.0, maximum_points)
    budget_seq = np.linspace(1.2, 5.0, maximum_points)
    return cost_ratio_seq, budget_seq


def run_single_instance(model_name, baseline, aux_args=None, maximum_points=5):
    """
    # TODO: think about (r, b) grid
    Returns List[Tuple[float, float, float]], cost ratio, budget, f(S)
    """
    model = model_constructor(model_name)
    triples = []
    cost_ratio_seq, budget_seq = generate_rb_grid(maximum_points)
    for cost_ratio in cost_ratio_seq:
        for budget in budget_seq:
            if cost_ratio >= budget:
                # invalid pair
                continue
            model.constraint.enforce_cost_ratio_and_budget(cost_ratio, budget)
            if aux_args is not None:
                res = baseline(model, aux_args)  # basline is the greedy aggregate function
            else:
                res = baseline(model)
            assert 'f(S)' in res, "f(S) not in result"
            fv = res['f(S)']
            triples.append((cost_ratio, budget, fv))
    return triples


def combine_precompute_results(precompute_results):
    """
    Combine precompute results into one result
    """
    density_res = precompute_results['density']
    marginal_gain_res = precompute_results['marginal_gain']
    res = []
    for left_triple, right_triple in zip(density_res, marginal_gain_res):
        assert left_triple[0] == right_triple[0] and left_triple[1] == right_triple[1], "cost_ratio and budget should be the same"
        res.append((left_triple[0], left_triple[1], max(left_triple[2], right_triple[2])))
    return res
    
def run_batch(model_name_lst: List[str] = ["MovieRecommendation", "ImageSum"], output_file: str = "output.txt", maximum_points=5):
    # put all data into one table, sequential execution
    rows = []
    for model in tqdm(model_name_lst, desc="Models"):
        precompute_result = {}
        for baseline_name, baseline_func in tqdm(baselines.items(), desc=f"{model} Baselines", leave=False):
            if baseline_name == "Aggregate":
                assert len(precompute_result) == 2, "Precompute results should contain density and marginal_gain"
                # run_single_instance(model, baseline_func, aux_args=precompute_result)
                res = combine_precompute_results(precompute_result)
            elif baseline_name == "Density":
                res = run_single_instance(model, baseline_func, maximum_points=maximum_points)
                precompute_result['density'] = res
            elif baseline_name == "Marginal gain":  
                res = run_single_instance(model, baseline_func, maximum_points=maximum_points)
                precompute_result['marginal_gain'] = res
            else:
                res = run_single_instance(model, baseline_func, maximum_points=maximum_points)
            # res contains list of [cost ratio, budget, f(S)]
            # expand into tuple with len 5
            res = [(model, baseline_name, *triple) for triple in res]
            rows.extend(res)

    # write to file
    with open(output_file, "w") as f:
        f.write("model_name,baseline_name,cost_ratio,budget,f(S)\n")
        for row in rows:
            f.write(f"{','.join(map(str, row))}\n")
    
def main():
    dump_dir = "result/CostInterpolation-20250211/performance"
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
        logging.info("Created directory: {} for dumping results.".format(dump_dir))
    parser = argparse.ArgumentParser(description="Run cost ratio interpolation experiments.")
    parser.add_argument("--maximum_points", type=int, default=5, help="Maximum number of points for dominance and non-dominance.")
    args = parser.parse_args()

    maximum_points = args.maximum_points
    output_file = os.path.join(dump_dir, f"output_{maximum_points}.txt")

    if maximum_points <= 5:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)



    model_name_lst=["MovieRecommendation", "ImageSum", "MaxFLP", "MaxRevenue"]
    run_batch(model_name_lst, output_file, maximum_points)

if __name__ == "__main__":
    main()