import time

from OptimalAlg import OptimalAlg
from base_task import BaseTask
from data_dependent_upperbound import marginal_delta_random_budget


class IDAstar(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.threshold = 0
        self.min_upper_bound = None
        self.bound = None
        self.h = None
        self.node_count = 0

    def f(self, s):
        return self.g(s) + self.h(s)

    def g(self, s):
        # print(f"s:{s}")
        return self.model.objective(list(s))

    def search(self, path):
        node = path[len(path) - 1]
        self.node_count += 1
        # if self.node_count % 100 == 0:
            # print(f"node count:{self.node_count}, min:{self.min_upper_bound}")

        f_value = self.f(node)
        if self.min_upper_bound is None or f_value < self.min_upper_bound:
            self.min_upper_bound = f_value

        if f_value < self.bound:
            return None, f_value
        if self.is_goal(node):
            return node, f_value
        max_f = 0
        for succ in self.successors(node):
            if succ not in path:
                path.append(node | {succ})
                found, t = self.search(path)
                if found is not None:
                    return found, t
                if t > max_f:
                    max_f = t
                path.pop(len(path)-1)

        return None, max_f

    def successors(self, s):
        s_cost = self.model.cost_of_set(s)
        ret = []
        for i in set(self.model.ground_set) - set(s):
            if s_cost + self.model.cost_of_singleton(i) <= self.model.budget:
                ret.append(i)
        return ret

    def is_goal(self, s):
        if self.model.objective(s) >= self.alpha * self.min_upper_bound:
            # print("test here")
            return True
        return False

    def h_ub0(self, s):
        delta, _ = marginal_delta_random_budget(set(s), set(self.model.ground_set) - set(s), self.model, budget=self.model.budget)
        return delta

    def build(self):
        self.min_upper_bound = None
        self.bound = 0
        self.h = self.h_ub0
        print()

    def optimize(self):
        root = set()
        self.bound = self.h(root)
        path = [root]
        self.node_count = 0
        start_time = time.time()
        while True:
            found, t = self.search(path)
            if found is not None:
                stop_time = time.time()
                ret = {
                    "S": found,
                    "f(S)": self.g(found),
                    "c(S)": self.model.cost_of_set(list(found)),
                    "node_count": self.node_count,
                    "time": stop_time - start_time
                }
                return ret
            else:
                self.bound = t
