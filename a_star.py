import time

from OptimalAlg import OptimalAlg
from base_task import BaseTask
from MaxHeap import MaxHeap, HeapObj
from data_dependent_upperbound import marginal_delta_version7, marginal_delta


class A_star(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

        self.closed_list = []
        self.heap = MaxHeap()
        self.f = None
        self.h = None
        self.alpha = 1

        self.E = set()

    def build(self):
        self.closed_list.clear()
        self.heap.clear()
        self.f = self.model.objective
        if self.opt == 'ub0':
            self.h = self.h_ub0
        elif self.opt == 'ub2':
            self.h = self.h_ub2

    # the heuristic function
    def h_ub0(self, S):
        delta, _ = marginal_delta(set(S), set(self.model.ground_set) - set(S), self.model)
        return delta

    # the heuristic function
    def h_ub2(self, S):
        delta, _ = marginal_delta_version7(set(S), set(self.model.ground_set) - set(S), self.model)
        return delta

    def g(self, S):
        return self.f(list(S)) + self.alpha * self.h(S)

    def greedy_with_base(self, S):
        model = self.model

        sol = S
        remaining_elements = set(model.ground_set) - S
        cur_cost = model.cost_of_set(list(S))

        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds
            assert u is not None
            if cur_cost + model.cost_of_singleton(u) <= model.budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += model.cost_of_singleton(u)

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if model.cost_of_singleton(v) + cur_cost > model.budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return sol

    def optimize(self):
        start_time = time.time()

        ret = {

        }

        s_star = set()
        s_star_v = self.model.objective(list(s_star))

        L = MaxHeap()
        L.push(HeapObj(s_star, self.g(s_star)))

        while L.size() > 0:
            obj = L.pop()
            s, g_s = obj.s, obj.v
            if g_s > s_star_v:
                s_plus = self.greedy_with_base(s)
                if self.model.objective(s_plus) > s_star_v:
                    s_star_v = self.model.objective(s_plus)
                    s_star = s_plus
                for ele in set(self.model.ground_set) - s:
                    t = s | {ele}
                    if self.model.cost_of_set(t) <= self.model.budget and self.g(t) >= s_star_v:
                        L.push(HeapObj(t, self.g(t)))

        stop_time = time.time()

        ret['S'] = s_star
        ret['c(S)'] = self.model.cost_of_set(s_star)
        ret['f(S)'] = s_star_v
        ret['time'] = stop_time - start_time

        return ret
