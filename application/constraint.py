from typing import List
import logging
logging.basicConfig(level=logging.DEBUG)

cardinality_id = 1
knapsack_id = 2

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
    def __init__(self, cost_obj: List[int], budget: float):
        super().__init__("knapsack constraint")
        self.id = knapsack_id
        assert min(cost_obj) >= 0, "Abort: Cost can't be negative"
        self.cost_func = cost_obj
        self.budget = budget
        self.budget_ratio = budget / sum(cost_obj)

        self.is_normalized = False

    def __init__(self, cost_obj: List[int], budget_ratio: float):
        logging.debug("Use budget ratio constructor")
        super().__init__("knapsack constraint")
        self.id = knapsack_id
        assert min(cost_obj) >= 0, "Abort: Cost can't be negative\n{}".format(min(cost_obj))
        self.cost_func = cost_obj
        assert 0.0 <= budget_ratio <= 1.0, "Budget ratio error: {}".format(budget_ratio)
        self.budget_ratio = budget_ratio
        self.budget = sum(cost_obj) * budget_ratio

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

    