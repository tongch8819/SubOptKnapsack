from application.movie_recommendation_knapsack import MovieRecommendationKnapsack
from algorithms.greedy import greedy_density_knapsack, greedy_marginal_gain_knapsack

import math
import numpy as np

def is_dominance_theoretical(cost_ratio: float, budget: float):
    assert cost_ratio <= budget, "Cost ratio should be less than budget"
    d = np.minimum(budget, cost_ratio * math.floor(budget))
    af_density = 1 - math.exp(- (budget - cost_ratio) / d)
    af_marginal_gain =  math.floor(cost_ratio / budget) / math.floor(budget)
    return af_density >= af_marginal_gain

def is_dominance_practical(density_res, marginal_gain_res):
    return density_res['f(S)'] >= marginal_gain_res['f(S)']

def main():
    # for budget_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
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

    print("Is density dominant? (Theorectical)", is_dominance_theoretical(cost_ratio, budget))
    print("Is density dominant? (Practical)", is_dominance_practical(density_res, marginal_gain_res))

if __name__ == "__main__":
    main()
