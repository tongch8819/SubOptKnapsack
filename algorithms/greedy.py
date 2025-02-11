from application.base_task import BaseTask
from application.constraint import is_knapsack_constraint
from copy import deepcopy
import numpy as np
import heapq



def greedy_knapsack(model: BaseTask, gain_func):
    """
    General greedy algorithm for knapsack problem with abstract gain function
    Inputs:
    - model: BaseTask object whose constraint is knapsack
    - gain_func: function that takes in model, element, and current solution, and returns the gain of adding the element to the solution
    """
    assert is_knapsack_constraint(model.constraint), "Greedy knapsack only works for knapsack constraint"
    budget = model.constraint.budget
    cost_func = model.constraint.cost_func

    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.0

    while remaining_elements:
        u = max(remaining_elements, key=lambda e: gain_func(model, e, list(sol)))
        if cur_cost + cost_func[u] <= budget:
            sol.add(u)
            cur_cost += cost_func[u]
        remaining_elements.remove(u)
        remaining_elements = {e for e in remaining_elements if cost_func[e] + cur_cost <= budget}

    return {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': cur_cost,
    }


def greedy_density_knapsack(model: BaseTask):
    """
    Use Density as the greedy function
    """
    return greedy_knapsack(model, lambda m, e, s: m.density(e, s))


def greedy_marginal_gain_knapsack(model: BaseTask):
    """
    Use Marginal Gain as the greedy function
    """
    return greedy_knapsack(model, lambda m, e, s: m.marginal_gain(e, s))

def greedy_aggregate(model: BaseTask, precompute_results=None): 
    """
    Use the aggregate function as the greedy function
    """
    if precompute_results is None:
        res_density = greedy_density_knapsack(model)
        res_marginal_gain = greedy_marginal_gain_knapsack(model)
    else:
        assert 'density' in precompute_results and 'marginal_gain' in precompute_results, "Precompute results should contain density and marginal_gain"
        res_density = precompute_results['density']
        res_marginal_gain = precompute_results['marginal_gain']
    assert 'f(S)' in res_density and 'f(S)' in res_marginal_gain, f"f(S) not in result: {res_density}, {res_marginal_gain}"
    if res_density['f(S)'] >= res_marginal_gain['f(S)']:
        return res_density
    else:
        return res_marginal_gain

def greedy_max(model: BaseTask):
    """
    GreedyMax algorithm from paper: "Bring Your Own Greedy"+Max: Near-Optimal $1/2$-Approximations for Submodular Knapsack
    """
    assert is_knapsack_constraint(model.constraint), "Greedy knapsack only works for knapsack constraint"
    G, S = set(), set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    while len(remaining_elements):
        # argmax marginal gain
        s, max_marginal_gain = None, -1
        for e in remaining_elements:
            mg = model.marginal_gain(e, G)
            if s is None or mg > max_marginal_gain:
                s, max_marginal_gain = e, mg
        assert s is not None
        # 
        tmp_G = deepcopy(G)
        tmp_G.add(s)
        if model.objective(S) < model.objective(tmp_G) and model.constraint.cost_of_set(tmp_G) <= model.constraint.budget:
            S = tmp_G
        # argmax density
        a, max_density = None, -1.
        for e in remaining_elements:
            # e is an object
            ds = model.density(e, G)
            if a is None or ds > max_density:
                a, max_density = e, ds
        assert a is not None
        if cur_cost + model.constraint.cost_of_singleton(a) <= model.constraint.budget:
            G.add(a)
            cur_cost += model.constraint.cost_of_singleton(a)

        remaining_elements.remove(a)
        # filter out violating elements
        to_remove = set()
        for v in remaining_elements:
            if model.constraint.cost_of_singleton(v) + cur_cost > model.constraint.budget:
                to_remove.add(v)
        remaining_elements -= to_remove

    S_fv = model.objective(S)
    G_fv = model.objective(G)
    if S_fv >= G_fv:
        res = {
            'S': S,
            'f(S)': S_fv,
            'c(S)': model.constraint.cost_of_set(S),
        }
    else:
        res = {
            'S': G,
            'f(S)': G_fv,
            'c(S)': model.constraint.cost_of_set(G),
        }
    return res

def modified_greedy(model: BaseTask):
    """
    Modified Greedy algorithm from paper: The budgeted maximum coverage problem
    """
    assert is_knapsack_constraint(model.constraint), "Greedy knapsack only works for knapsack constraint"

    res = greedy_density_knapsack(model)
    # argmax singleton
    optimal_singleton, optimal_singleton_value = None, -1
    for e in model.ground_set:
        if optimal_singleton is None or model.objective({e}) > optimal_singleton_value:
            optimal_singleton, optimal_singleton_value = e, model.objective({e})
    singleton_solution = {
        'S': {optimal_singleton},
        'f(S)': optimal_singleton_value,
        'c(S)': model.constraint.cost_of_singleton(optimal_singleton),
    }

    # return the best between
    if res['f(S)'] >= singleton_solution['f(S)']:
        return res
    else:
        return singleton_solution
    

def is_dominance_theoretical(cost_ratio: float, budget: float):
    """
    Check if density knapsack greedy is dominant over marginal gain knapsack greedy theoretically
    See more details in cost ratio interpolation paper.
    """
    assert cost_ratio <= budget, "Cost ratio should be less than budget"
    d = np.minimum(budget, cost_ratio * np.floor(budget))
    af_density = 1 - np.exp(- (budget - cost_ratio) / d)
    C = 1 - np.exp(-1)
    af_marginal_gain = C * np.floor(budget / cost_ratio) / np.floor(budget)
    # logging.debug(f"af_density: {af_density}, af_marginal_gain: {af_marginal_gain}")
    return af_density >= af_marginal_gain


class MaxHeap:
    def __init__(self, costs):
        """Initialize the max heap with (negated cost, index) for max heap behavior."""
        self.max_heap = [(-cost, i) for i, cost in enumerate(costs)]
        heapq.heapify(self.max_heap)  # Convert list to a valid heap
        self.removed = set()  # Keep track of removed indices

    def remove(self, idx):
        """Mark an index as removed (lazy deletion)."""
        self.removed.add(idx)

    def get_max(self):
        """Return the maximum cost in the heap, ensuring it's not removed."""
        while self.max_heap and self.max_heap[0][1] in self.removed:
            heapq.heappop(self.max_heap)  # Remove invalid elements
        return -self.max_heap[0][0] if self.max_heap else None  # Return max cost

def adaptive_greedy(model: BaseTask):
    """
    Adaptive greedy algorithm: at each iteration step, choose the best between density and marginal gain adaptively
    """
    assert is_knapsack_constraint(model.constraint), "Greedy knapsack only works for knapsack constraint"
    budget = model.constraint.budget
    cost_func = model.constraint.cost_func

    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.0

    cost_ratio_heap = MaxHeap([cost_func[e] for e in model.ground_set])
    residual_cost_ratio = cost_ratio_heap.get_max()
    residual_budget = budget 

    while remaining_elements:
        gain_func = (lambda m, e, s: m.density(e, s)) if is_dominance_theoretical(residual_cost_ratio, residual_budget) else (lambda m, e, s: m.marginal_gain(e, s))
        
        optimal_element = max(remaining_elements, key=lambda e: gain_func(model, e, list(sol)))
        
        if cur_cost + cost_func[optimal_element] <= budget:
            sol.add(optimal_element)
            cur_cost += cost_func[optimal_element]

        cost_ratio_heap.remove(optimal_element)
        remaining_elements.remove(optimal_element)
        remaining_elements = {e for e in remaining_elements if cost_func[e] + cur_cost <= budget}
        
        residual_cost_ratio = cost_ratio_heap.get_max()
        residual_budget = budget - cur_cost

    return {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': cur_cost,
    }
