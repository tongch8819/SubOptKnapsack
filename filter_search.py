import time

from OptimalAlg import OptimalAlg
from base_task import BaseTask
from MaxHeap import MaxHeap, HeapObj
from data_dependent_upperbound import marginal_delta_version7, marginal_delta


class FS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

        self.closed_list = []
        self.heap = MaxHeap()
        self.f = None
        self.h = None

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

    def optimize(self):
        start_time = time.time()

        ret = {

        }

        s = None
        self.heap.push(HeapObj(set(), self.g({})))
        while self.heap.size() > 0:
            obj = self.heap.pop()
            s, v = obj.s, obj.v
            if self.h(s) == 0:
                break
            if s not in self.closed_list:
                self.closed_list.append(s)

            for ele in set(self.model.ground_set) - s:
                s_plus = s | {ele}
                if self.model.cost_of_set(s_plus) <= self.model.budget:
                    self.heap.push(HeapObj(s_plus, self.g(s_plus)))

        stop_time = time.time()

        ret['S'] = s
        ret['c(S)'] = self.model.cost_of_set(s)
        ret['f(S)'] = self.model.objective(s)
        ret['time'] = stop_time - start_time

        return ret
