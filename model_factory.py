from base_task import BaseTask
from facebook_graph_coverage import FacebookGraphCoverage
from feature_selection import AdultIncomeFeatureSelection
from influence_maximization import YoutubeCoverage
from revenue_max import CalTechMaximization

def model_factory(task:str, n, seed, budget, knap = True, cm = 'normal', enable_packing = False, constraint_count = 4) -> BaseTask:
    model = None
    if task == "adult":
        model = AdultIncomeFeatureSelection(0, n, "./dataset/adult-income", seed=seed, sample_count=100, knapsack=knap, cost_mode=cm, construct_graph=True, enable_packing=enable_packing, constraint_count=constraint_count)
    elif task == "caltech":
        model = CalTechMaximization(0, n, "./dataset/caltech", seed=seed, knapsack=knap, prepare_max_pair=False,
                                    cost_mode=cm, print_curvature=False, construct_graph=True, enable_packing=enable_packing, constraint_count=constraint_count)
    elif task == "facebook":
        model = FacebookGraphCoverage(
            budget=0, n=n, seed=seed, graph_path="./dataset/facebook", knapsack=knap, prepare_max_pair=False,
            print_curvature=False, cost_mode=cm, construct_graph=True, enable_packing=enable_packing, constraint_count = constraint_count)
    elif task == "youtube":
        model = YoutubeCoverage(0, n, "./dataset/com-youtube", seed=seed, knapsack=knap, cost_mode=cm,
                                prepare_max_pair=False, print_curvature=False, construct_graph=True, enable_packing=enable_packing, constraint_count = constraint_count)

    model.budget = budget

    return model