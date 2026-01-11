from MaxHeap import HeapObj, BaseHeapObj
from base_task import BaseTask
from data_dependent_upperbound import marginal_delta_dom_random_budget, marginal_delta_version7_random_budget, \
    marginal_delta_random_budget
from optimizer import DominantOptimizer


class OptimalAlg:
    def __init__(self, model: BaseTask):
        self.model = model
        self.alpha = 0.8
        self.opt = None
        pass

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setOpt(self, opt):
        self.opt = opt

    def build(self):
        pass

    def optimize(self):
        pass

    def set_h(self, heuristic):
        if heuristic == 'ub0':
            self.inner_h = self.h_ub0
            self.lbd = self.lbd0
        elif heuristic == 'ub2':
            self.inner_h = self.h_ub2
            self.lbd = self.lbd2
        elif heuristic == 'dom':
            self.inner_h = self.h_dom
            self.lbd = self.lbd_dom

    def density_for_set(self, n):
        if self.model.cost_of_set(list(n)) == 0:
            return 0
        return self.g(n) / self.model.cost_of_set(list(n))

    def set_d(self, sorting):
        if sorting == 'g':
            self.d = self.g
        elif sorting == 'b':
            self.d = lambda x: self.model.cost_of_set(list(x))
        elif sorting == 'd':
            self.d = self.density_for_set

    def g(self, n):
        if isinstance(n, BaseHeapObj):
            return self.model.objective(list(n.s))

        return self.model.objective(list(n))

    def is_on_the_edge(self, node):
        candidate = node.candidate
        for ele in candidate:
            if self.model.cost_of_singleton(ele) <= node.budget:
                return False
        return True

    def h(self, n):
        if self.is_on_the_edge(n):
            return 0
        return self.inner_h(n)

    def f_with_alpha(self, n):
        return self.g(n) + self.alpha * self.h(n)

    def f_without_alpha(self, n):
        return self.g(n) + self.h(n)

    def h_ub0(self, n):
        delta, _ = marginal_delta_random_budget(set(n.s), set(n.candidate), self.model,
                                                budget=n.budget)
        return delta

    def h_ub2(self, n):
        delta, _ = marginal_delta_version7_random_budget(set(n.s), set(n.candidate), self.model,
                                                         budget=n.budget)
        return delta

    def h_dom(self, n):
        delta, _ = marginal_delta_dom_random_budget(set(n.s), set(n.candidate), self.model,
                                                    budget=n.budget)
        return delta

    def lbd0(self, base, candidate, budget):
        delta, _ = marginal_delta_random_budget(set(base), set(candidate), self.model,
                                                budget=budget)
        return delta

    def lbd2(self, base, candidate, budget):
        delta, _ = marginal_delta_version7_random_budget(set(base), set(candidate), self.model,
                                                         budget=budget)
        return delta

    def lbd_dom(self, base, budget):
        delta, _ = marginal_delta_dom_random_budget(set(base), set(self.model.ground_set) - set(base), self.model,
                                                    budget=budget)
        return delta

    def h_ub4(self, n):
        opt = DominantOptimizer()
        opt.setModel(self.model)
        opt.setBase(n)
        opt.build()
        delta = opt.optimize()['delta']

        return delta