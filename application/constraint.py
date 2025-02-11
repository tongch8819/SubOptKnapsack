from typing import List
import logging

cardinality_id = 1
knapsack_id = 2

def is_knapsack_constraint(constraint):
    return constraint.id == knapsack_id

class Constraint:
    def __init__(self, name="general constraint"):
        self.id = 0
        self.name = name

class CardinalityConstraint(Constraint):
    def __init__(self, k: int):
        super().__init__("cardinality constraint")
        self.size = k
        self.id = cardinality_id

class KnapsackConstraint(Constraint):
    def __init__(self, cost_obj: List[int], budget: float = None, budget_ratio: float = None):
        super().__init__("knapsack constraint")
        self.id = knapsack_id
        assert min(cost_obj) >= 0, "Abort: Cost can't be negative"
        self.cost_func = cost_obj

        if budget is not None and budget_ratio is not None:
            raise ValueError("Either budget or budget_ratio must be provided, not both")

        if budget is not None:
            self.budget = budget
            self.budget_ratio = budget / sum(cost_obj)
        elif budget_ratio is not None:
            # logging.debug("Use budget ratio constructor")
            assert 0.0 <= budget_ratio <= 1.0, "Budget ratio error: {}".format(budget_ratio)
            self.budget_ratio = budget_ratio
            self.budget = sum(cost_obj) * budget_ratio
        else:
            raise ValueError("Either budget or budget_ratio must be provided")

        self.is_normalized = False

    def cost_of_set(self, S: List[int]):
        return sum(self.cost_func[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.cost_func), "Singleton: {}".format(singleton)
        return self.cost_func[singleton]

    def normalize(self):
        if self.is_normalized:
            return
        min_cost = min(self.cost_func)
        assert min_cost > 0, "Min cost: {}".format(min_cost)
        self.cost_func = [x / min_cost for x in self.cost_func]
        self.budget /= min_cost
        assert abs(self.budget / sum(self.cost_func) - self.budget_ratio) <= 1e-4, "Budget ratio error: {}, {}, {}".format(self.budget_ratio, self.budget, sum(self.cost_func))
        self.budget_ratio = self.budget / sum(self.cost_func)
        self.is_normalized = True

    def enforce_cost_ratio_and_budget(self, cost_ratio: float, budget: float):
        """
        # Linear scaling formula in LaTeX:
        # c'_i = \\frac{(c_i - c_{\min})}{c_{\max} - c_{\min}} (r - 1) c_{\min} + c_{\min}
        """
        self.normalize()
        assert cost_ratio <= budget, "Cost ratio should be less than budget"
        c_min = min(self.cost_func)
        c_max = max(self.cost_func)
        diff = c_max - c_min

        k = (cost_ratio - 1) * c_min / diff
        b = - c_min * (cost_ratio - 1) * c_min / diff + c_min

        self.cost_func = [ k * ci + b for ci in self.cost_func]
        # self.budget = k * self.budget + b
        self.budget = budget
        
    def show_stat(self):
        cmin, cmax = min(self.cost_func), max(self.cost_func)
        budget_ratio = self.budget / sum(self.cost_func)
        return f"Min cost: {cmin}, Max cost: {cmax}, Budget ratio: {budget_ratio}, Budget: {self.budget}"   