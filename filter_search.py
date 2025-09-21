import time
from functools import total_ordering

from OptimalAlg import OptimalAlg
from base_task import BaseTask
from MaxHeap import MaxHeap, HeapObj
from data_dependent_upperbound import marginal_delta_version7, marginal_delta, marginal_delta_m, marginal_delta_m_acc, \
    marginal_delta_random_budget, marginal_delta_version7_random_budget, marginal_delta_m_acc_random_budget


@total_ordering
class AugmentedValue:
    def __init__(self, av, v):
        self.av = av
        self.v = v

    def __eq__(self, other):
        return self.av == other.av and self.v == other.v

    def __lt__(self, other):
        if self.av < other.av:
            return True
        if self.av == other.av and self.v < other.v:
            return True
        return False


# class FS(OptimalAlg):
#     def __init__(self, model: BaseTask):
#         super().__init__(model)
#
#         self.closed_list = []
#         self.heap = MaxHeap()
#         self.f = None
#         self.h = None
#
#         self.lbd = -1
#         self.augmentation = False
#
#         self.E = set()
#
#     def build(self):
#         self.closed_list.clear()
#         self.heap.clear()
#         self.lbd = -1
#         self.augmentation = False
#
#         self.f = self.model.objective
#         if self.opt == 'ub0':
#             self.h = self.h_ub0
#         elif self.opt == 'ub1':
#             self.h = self.h_ub1
#         elif self.opt == 'ub2':
#             self.h = self.h_ub2
#         elif self.opt == 'ub0+':
#             self.h = self.h_ub0
#             self.augmentation = True
#         elif self.opt == 'ub1+':
#             self.h = self.h_ub1
#             self.augmentation = True
#         elif self.opt == 'ub2+':
#             self.h = self.h_ub2
#             self.augmentation = True
#
#     # the heuristic function
#     def h_ub0(self, S):
#         delta, _ = marginal_delta_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     # the heuristic function
#     def h_ub2(self, S):
#         delta, _ = marginal_delta_version7_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     # the heuristic function
#     def h_ub1(self, S):
#         delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
#         # delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     def is_on_the_edge(self, S):
#         base_cost = self.model.cost_of_set(S)
#         for ele in set(self.model.ground_set) - S:
#             if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
#                 return False
#         return True
#
#     def g(self, S):
#         if self.is_on_the_edge(S):
#             return self.f(list(S))
#
#         current_h = self.f(list(S)) + self.alpha * self.h(S)
#
#         if self.augmentation:
#             if self.lbd < 0 or current_h < self.lbd:
#                 self.lbd = current_h
#             else:
#                 current_h = self.lbd
#
#         return current_h
#
#     def optimize(self):
#         start_time = time.time()
#
#         ret = {
#         }
#
#         s = None
#
#         node_count = 0
#
#         self.heap.push(HeapObj(set(), self.g(set())))
#         while self.heap.size() > 0:
#             obj = self.heap.pop()
#             s, v = obj.s, obj.v
#
#             # print(f"s:{s}, v:{v}, lbd:{self.lbd}")
#
#             node_count += 1
#
#             if self.is_on_the_edge(s):
#                 stop_time = time.time()
#                 ret['S'] = s
#                 ret['c(S)'] = self.model.cost_of_set(s)
#                 ret['f(S)'] = self.model.objective(s)
#                 ret['time'] = stop_time - start_time
#                 ret['node_count'] = node_count
#                 return ret
#
#             if s not in self.closed_list:
#                 self.closed_list.append(s)
#
#             for ele in set(self.model.ground_set) - s:
#                 s_plus = s | {ele}
#                 if self.model.cost_of_set(s_plus) <= self.model.budget:
#                     self.heap.push(HeapObj(s_plus, self.g(s_plus)))
#
#         stop_time = time.time()
#         ret['S'] = s
#         ret['c(S)'] = self.model.cost_of_set(s)
#         ret['f(S)'] = self.model.objective(s)
#         ret['time'] = stop_time - start_time
#         ret['node_count'] = node_count
#         print(f"return from fallback")
#
#         return ret

class FS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

        self.closed_list = []
        self.heap = MaxHeap()
        self.f = None
        self.h = None

        self.lbd = -1
        self.augmentation = False

        self.E = set()

    def build(self):
        self.closed_list.clear()
        self.heap.clear()
        self.lbd = -1
        self.augmentation = False

        self.f = self.model.objective
        if self.opt == 'ub0':
            self.h = self.h_ub0
        elif self.opt == 'ub1':
            self.h = self.h_ub1
        elif self.opt == 'ub2':
            self.h = self.h_ub2
        elif self.opt == 'ub0+':
            self.h = self.h_ub0
            self.augmentation = True
        elif self.opt == 'ub1+':
            self.h = self.h_ub1
            self.augmentation = True
        elif self.opt == 'ub2+':
            self.h = self.h_ub2
            self.augmentation = True

    # the heuristic function
    def h_ub0(self, S):
        delta, _ = marginal_delta_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub2(self, S):
        delta, _ = marginal_delta_version7_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub1(self, S):
        # delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
        delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    def is_on_the_edge(self, S):
        base_cost = self.model.cost_of_set(S)
        for ele in set(self.model.ground_set) - S:
            if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
                return False
        return True

    def g(self, S):
        if self.is_on_the_edge(S):
            return self.f(list(S)), self.f(list(S))

        current_h = self.f(list(S)) + self.alpha * self.h(S)
        normal_h = current_h

        if self.augmentation:
            if self.lbd < 0 or current_h < self.lbd:
                self.lbd = current_h
            else:
                current_h = self.lbd

        return current_h, normal_h

    def push_heap(self, S):
        aug_v, v = self.g(S)
        t = AugmentedValue(aug_v, v)
        self.heap.push(HeapObj(S, t))

    def optimize(self):
        start_time = time.time()

        ret = {
        }

        s = None

        node_count = 0

        self.push_heap(set())
        while self.heap.size() > 0:
            obj = self.heap.pop()
            s, v = obj.s, obj.v

            # print(f"s:{s}, v:{v}, lbd:{self.lbd}")

            node_count += 1

            if self.is_on_the_edge(s):
                stop_time = time.time()
                ret['S'] = s
                ret['c(S)'] = self.model.cost_of_set(s)
                ret['f(S)'] = self.model.objective(s)
                ret['time'] = stop_time - start_time
                ret['node_count'] = node_count
                return ret

            if s not in self.closed_list:
                self.closed_list.append(s)

            for ele in set(self.model.ground_set) - s:
                s_plus = s | {ele}
                if self.model.cost_of_set(s_plus) <= self.model.budget:
                    self.push_heap(s_plus)

        stop_time = time.time()
        ret['S'] = s
        ret['c(S)'] = self.model.cost_of_set(s)
        ret['f(S)'] = self.model.objective(s)
        ret['time'] = stop_time - start_time
        ret['node_count'] = node_count
        print(f"return from fallback")

        return ret


class AugmentedFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

        self.closed_list = []
        self.heap = MaxHeap()

        self.g = None
        self.inner_h = None

        self.lbd = -1
        self.augmentation = False

        self.E = set()

    def build(self):
        self.closed_list.clear()
        self.heap.clear()
        self.lbd = -1

        self.g = self.model.objective

        if self.opt == 'ub0':
            self.inner_h = self.h_ub0
        elif self.opt == 'ub1':
            self.inner_h = self.h_ub1
        elif self.opt == 'ub2':
            self.inner_h = self.h_ub2

    # the heuristic function
    def h_ub0(self, S):
        delta, _ = marginal_delta_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub2(self, S):
        delta, _ = marginal_delta_version7_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub1(self, S):
        # delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
        delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    def is_on_the_edge(self, S):
        base_cost = self.model.cost_of_set(S)
        for ele in set(self.model.ground_set) - S:
            if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
                return False
        return True

    def h(self, S):
        if self.is_on_the_edge(S):
            return 0
        return self.inner_h(S)

    def f(self, node: HeapObj):
        return min(self.g(node.s) + self.alpha * self.h(node.s), node.lbd)

    def push_heap(self, S, parent_lbd):
        current_lbd = min(self.g(S) + self.alpha * self.h(S), parent_lbd)

        t = AugmentedValue(current_lbd, self.g(S))

        self.heap.push(HeapObj(S, t, current_lbd))

    def optimize(self):
        start_time = time.time()

        ret = {
        }

        s = None

        node_count = 0

        self.push_heap(set(), self.alpha * self.h(set()))

        while self.heap.size() > 0:
            obj = self.heap.pop()
            s, v, parent_lbd = obj.s, obj.v, obj.lbd

            # print(f"s:{s}, v:{v}, lbd:{self.lbd}")

            node_count += 1

            if self.is_on_the_edge(s):
                stop_time = time.time()
                ret['S'] = s
                ret['c(S)'] = self.model.cost_of_set(s)
                ret['f(S)'] = self.model.objective(s)
                ret['time'] = stop_time - start_time
                ret['node_count'] = node_count
                return ret

            if s not in self.closed_list:
                self.closed_list.append(s)

            for ele in set(self.model.ground_set) - s:
                s_plus = s | {ele}
                if self.model.cost_of_set(s_plus) <= self.model.budget:
                    self.push_heap(s_plus, parent_lbd)

        stop_time = time.time()
        ret['S'] = s
        ret['c(S)'] = self.model.cost_of_set(s)
        ret['f(S)'] = self.model.objective(s)
        ret['time'] = stop_time - start_time
        ret['node_count'] = node_count
        print(f"return from fallback")

        return ret