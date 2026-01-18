import copy
import time
from functools import total_ordering

from OptimalAlg import OptimalAlg
from base_task import BaseTask
from MaxHeap import MaxHeap, HeapObj, EfficientBFSHeapObj, BranchAndBoundNode, SimpleMaxHeap, BaseHeapObj
from data_dependent_upperbound import marginal_delta_version7, marginal_delta, marginal_delta_m, marginal_delta_m_acc, \
    marginal_delta_random_budget, marginal_delta_version7_random_budget, marginal_delta_m_acc_random_budget, \
    marginal_delta_dom_random_budget
from optimizer import DominantOptimizer


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


@total_ordering
class AugmentedFSValue:
    def __init__(self, inner_v, lbd_v, sort_v):
        self.inner_v = inner_v
        self.lbd_v = lbd_v
        self.sort_v = sort_v

    def __eq__(self, other):
        return self.lbd_v == other.lbd_v and self.sort_v == other.sort_v

    def __lt__(self, other):
        if self.lbd_v < other.lbd_v:
            return True
        if self.lbd_v == other.lbd_v and self.sort_v < other.sort_v:
            return True
        return False


@total_ordering
class RefinedBFSValue:
    def __init__(self, inner_v, lbd_v, sort_v):
        self.inner_v = inner_v
        self.lbd_v = lbd_v
        self.sort_v = sort_v

    def __eq__(self, other):
        return self.lbd_v == other.lbd_v and self.sort_v == other.sort_v

    def __lt__(self, other):
        if self.lbd_v < other.lbd_v:
            return True
        if self.lbd_v == other.lbd_v and self.sort_v < other.sort_v:
            return True
        return False

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
        delta, _ = marginal_delta_random_budget(set(S), set(self.model.ground_set) - set(S), self.model,
                                                budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub2(self, S):
        delta, _ = marginal_delta_version7_random_budget(set(S), set(self.model.ground_set) - set(S), self.model,
                                                         budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub1(self, S):
        # delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
        delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model,
                                                      budget=self.model.budget - self.model.cost_of_set(S))
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
        explored_node_count = 0

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
                ret['explored_node_count'] = explored_node_count
                return ret

            if s not in self.closed_list:
                self.closed_list.append(s)

                for ele in set(self.model.ground_set) - s:
                    s_plus = s | {ele}
                    if self.model.cost_of_set(s_plus) <= self.model.budget:
                        explored_node_count += 1
                        self.push_heap(s_plus)

        stop_time = time.time()
        ret['S'] = s
        ret['c(S)'] = self.model.cost_of_set(s)
        ret['f(S)'] = self.model.objective(s)
        ret['time'] = stop_time - start_time
        ret['node_count'] = node_count
        ret['explored_node_count'] = explored_node_count
        print(f"return from fallback")

        return ret


# class AugmentedFS(OptimalAlg):
#     def __init__(self, model: BaseTask):
#         super().__init__(model)
#
#         self.closed_list = []
#         self.heap = MaxHeap()
#
#         self.g = None
#         self.inner_h = None
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
#
#         self.g = self.model.objective
#
#         if self.opt == 'ub0':
#             self.inner_h = self.h_ub0
#         elif self.opt == 'ub1':
#             self.inner_h = self.h_ub1
#         elif self.opt == 'ub2':
#             self.inner_h = self.h_ub2
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
#         # delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
#         delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
#         return delta
#
#     def is_on_the_edge(self, S):
#         base_cost = self.model.cost_of_set(S)
#         for ele in set(self.model.ground_set) - S:
#             if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
#                 return False
#         return True
#
#     def h(self, S):
#         if self.is_on_the_edge(S):
#             return 0
#         return self.inner_h(S)
#
#     def f(self, node: HeapObj):
#         return min(self.g(node.s) + self.alpha * self.h(node.s), node.lbd)
#
#     def push_heap(self, S, parent_lbd):
#         current_lbd = min(self.g(S) + self.alpha * self.h(S), parent_lbd)
#
#         t = AugmentedValue(current_lbd, self.g(S))
#
#         self.heap.push(HeapObj(S, t, current_lbd))
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
#         self.push_heap(set(), self.alpha * self.h(set()))
#
#         while self.heap.size() > 0:
#             obj = self.heap.pop()
#             s, v, parent_lbd = obj.s, obj.v, obj.lbd
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
#                 for ele in set(self.model.ground_set) - s:
#                     s_plus = s | {ele}
#                     if self.model.cost_of_set(s_plus) <= self.model.budget:
#                         self.push_heap(s_plus, parent_lbd)
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

# focal filter search
# class FFS(OptimalAlg):
#     def __init__(self, model: BaseTask):
#         super().__init__(model)
#
#         self.closed_list = []
#         self.heap = MaxHeap()
#         self.focal = []
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
#         self.focal.clear()
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
#         # delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
#         delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
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
#             return self.f(list(S)), self.f(list(S))
#
#         current_h = self.f(list(S)) + self.alpha * self.h(S)
#         normal_h = current_h
#
#         if self.augmentation:
#             if self.lbd < 0 or current_h < self.lbd:
#                 self.lbd = current_h
#             else:
#                 current_h = self.lbd
#
#         return current_h, normal_h
#
#     def push_heap(self, S):
#         aug_v, v = self.g(S)
#         t = AugmentedValue(aug_v, v)
#         self.heap.push(HeapObj(S, t))
#
#     def wrap(self, s):
#         aug_v, v = self.g(s)
#         t = AugmentedValue(aug_v, v)
#         obj = HeapObj(s, t)
#         return obj
#
#     def max_focal(self):
#         max_obj = None
#         for obj in self.focal:
#             if max_obj is None or max_obj < obj:
#                 max_obj = obj
#         return max_obj
#
#     def update_upper_bound(self, old_bound, new_bound):
#         for obj in self.heap.h:
#             if old_bound > obj.v.v >= new_bound and obj not in self.focal:
#                 self.focal.append(obj)
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
#         explored_node_count = 0
#
#         obj = self.wrap(set())
#         self.heap.push(obj)
#         self.focal.append(obj)
#         f_max = 0
#
#         while self.heap.size() > 0:
#             obj = self.max_focal()
#             f_max = self.heap.top().v
#
#             self.focal.remove(obj)
#             self.heap.remove(obj)
#
#             s, v = obj.s, obj.v
#             # print(f"s:{s}, v:{v}, lbd:{self.lbd}")
#             node_count += 1
#             if self.is_on_the_edge(s):
#                 stop_time = time.time()
#                 ret['S'] = s
#                 ret['c(S)'] = self.model.cost_of_set(s)
#                 ret['f(S)'] = self.model.objective(s)
#                 ret['time'] = stop_time - start_time
#                 ret['node_count'] = node_count
#                 ret['explored_node_count'] = explored_node_count
#                 return ret
#
#             if s not in self.closed_list:
#                 self.closed_list.append(s)
#
#                 for ele in set(self.model.ground_set) - s:
#                     s_plus = s | {ele}
#                     if self.model.cost_of_set(s_plus) <= self.model.budget:
#                         explored_node_count += 1
#                         obj = self.wrap(s_plus)
#                         self.heap.push(obj)
#                         if obj.v.v >= self.alpha * f_max.v:
#                             self.focal.append(obj)
#
#             if self.heap.size() > 0 and f_max > self.heap.top().v:
#                 self.update_upper_bound(self.alpha * f_max.v, self.alpha * self.heap.top().v.v)
#
#
#         stop_time = time.time()
#         ret['S'] = s
#         ret['c(S)'] = self.model.cost_of_set(s)
#         ret['f(S)'] = self.model.objective(s)
#         ret['time'] = stop_time - start_time
#         ret['node_count'] = node_count
#         ret['explored_node_count'] = explored_node_count
#         print(f"return from fallback")
#
#         return ret
#

class AugmentedFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = MaxHeap()
        self.inner_h = None
        self.d = None

    def build(self):
        self.max_heap.clear()

    def set_h(self, heuristic):
        if heuristic == 'ub0':
            self.inner_h = self.h_ub0
        elif heuristic == 'ub2':
            self.inner_h = self.h_ub2
        elif heuristic == 'ub4':
            self.inner_h = self.h_ub4

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
        return self.model.objective(list(n))

    def is_on_the_edge(self, n):
        base_cost = self.model.cost_of_set(n)
        for ele in set(self.model.ground_set) - set(n):
            if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
                return False
        return True

    def h(self, n):
        if self.is_on_the_edge(n):
            return 0
        return self.inner_h(n)

    def f(self, n):
        return self.g(n) + self.alpha * self.h(n)

    def h_ub0(self, n):
        delta, _ = marginal_delta_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
                                                budget=self.model.budget - self.model.cost_of_set(n))
        return delta

    def h_ub2(self, n):
        delta, _ = marginal_delta_version7_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
                                                         budget=self.model.budget - self.model.cost_of_set(n))
        return delta

    def h_ub4(self, n):
        opt = DominantOptimizer()
        opt.setModel(self.model)
        opt.setBase(n)
        opt.build()
        delta = opt.optimize()['delta']

        return delta

    def push_heap(self, n, inherited_value):
        max_value = 0
        if len(n) > 0:
            max_value = max(n)

        v = AugmentedFSValue(self.f(n), inherited_value, self.d(n), min(self.f(n), inherited_value), max_value)
        node = HeapObj(n, v)
        self.max_heap.push(node)

    def optimize(self):
        start_time = time.time()
        root = []
        self.push_heap(root, self.f(root))

        sol = None
        node_count = 0
        while self.max_heap.size() > 0:
            node = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v
            max_idx = v.max_idx
            if self.h(s) == 0:
                sol = s
                break

            for i in set(self.model.ground_set) - set(s):
                if i > max_idx and self.model.cost_of_set(s) + self.model.cost_of_singleton(i) <= self.model.budget:
                    inherited_value = min(self.f(list(set(s) | {i})), v.inherited_v)
                    self.push_heap(list(set(s) | {i}), inherited_value)

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count}

        return ret


class BestAugmentedFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = MaxHeap()
        self.inner_h = None
        self.f = None
        self.d = None
        self.use_alpha = False

    def build(self):
        self.max_heap.clear()
        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def set_h(self, heuristic):
        if heuristic == 'ub0':
            self.inner_h = self.h_ub0
        elif heuristic == 'ub2':
            self.inner_h = self.h_ub2
        elif heuristic == 'ub4':
            self.inner_h = self.h_ub4

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
        return self.model.objective(list(n))

    def is_on_the_edge(self, n):
        base_cost = self.model.cost_of_set(n)
        for ele in set(self.model.ground_set) - set(n):
            if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
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
        delta, _ = marginal_delta_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
                                                budget=self.model.budget - self.model.cost_of_set(n))
        return delta

    def h_ub2(self, n):
        delta, _ = marginal_delta_version7_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
                                                         budget=self.model.budget - self.model.cost_of_set(n))
        return delta

    def h_ub4(self, n):
        opt = DominantOptimizer()
        opt.setModel(self.model)
        opt.setBase(n)
        opt.build()
        delta = opt.optimize()['delta']

        return delta

    def push_heap(self, n, inherited_value):
        max_value = 0
        if len(n) > 0:
            max_value = max(n)

        v = AugmentedFSValue(self.f(n), inherited_value, self.d(n), min(self.f(n), inherited_value), max_value)
        node = HeapObj(n, v)
        self.max_heap.push(node)

    def greedy_add(self, base):
        sol = set(base)
        remaining_elements = set(self.model.ground_set) - set(base)
        cur_cost = self.model.cost_of_set(list(sol))

        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds
            assert u is not None
            if cur_cost + self.model.cost_of_singleton(u) <= self.model.budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > self.model.budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        # find the maximum singleton
        v_star, v_star_fv = None, float('-inf')
        for e in set(self.model.ground_set) - set(base):
            if self.model.cost_of_singleton(e) > self.model.budget - self.model.cost_of_set(list(base)):
                # filter out singleton whose cost is larger than budget
                continue
            fv = self.model.objective(list(set(base) | {e}))
            if fv > v_star_fv:
                v_star, v_star_fv = e, fv

        sol_fv = self.model.objective(list(sol))

        if v_star_fv > sol_fv:
            return list(set(base) | {v_star})
        else:
            return list(sol)

    def optimize(self):
        start_time = time.time()
        root = []
        self.push_heap(root, self.f(root))

        g_upper = self.f(root)
        s_max = self.greedy_add(root)
        s_max_v = self.g(s_max)
        sol = s_max

        node_count = 0
        open_list_count = 1

        while self.max_heap.size() > 0:
            node = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v
            max_idx = v.max_idx

            g_upper = min(g_upper, v.outlook_v)

            s_final = self.greedy_add(s)
            if self.g(s_final) > self.g(s_max):
                s_max = s_final
                s_max_v = self.g(s_max)

            if self.use_alpha:
                if self.g(s_max) >= g_upper:
                    sol = s_max
                    break
            else:
                if self.g(s_max) >= self.alpha * g_upper:
                    sol = s_max
                    break

            for i in set(self.model.ground_set) - set(s):
                if i > max_idx and self.model.cost_of_set(s) + self.model.cost_of_singleton(i) <= self.model.budget:
                    final_v = self.f_without_alpha(list(set(s) | {i}))
                    if final_v >= s_max_v:
                        inherited_value = min(self.f(list(set(s) | {i})), v.inherited_v)
                        self.push_heap(list(set(s) | {i}), inherited_value)
                        open_list_count += 1

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count}

        return ret


class BestAugmentedMoreFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = MaxHeap()
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0

    def build(self):
        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    # def set_h(self, heuristic):
    #     if heuristic == 'ub0':
    #         self.inner_h = self.h_ub0
    #         self.lbd = self.lbd0
    #     elif heuristic == 'ub2':
    #         self.inner_h = self.h_ub2
    #         self.lbd = self.lbd2
    #     elif heuristic == 'dom':
    #         self.inner_h = self.h_dom
    #         self.lbd = self.lbd_dom
    #
    # def density_for_set(self, n):
    #     if self.model.cost_of_set(list(n)) == 0:
    #         return 0
    #     return self.g(n) / self.model.cost_of_set(list(n))
    #
    # def set_d(self, sorting):
    #     if sorting == 'g':
    #         self.d = self.g
    #     elif sorting == 'b':
    #         self.d = lambda x: self.model.cost_of_set(list(x))
    #     elif sorting == 'd':
    #         self.d = self.density_for_set
    #
    # def g(self, n):
    #     return self.model.objective(list(n))
    #
    # def is_on_the_edge(self, n):
    #     base_cost = self.model.cost_of_set(n)
    #     for ele in set(self.model.ground_set) - set(n):
    #         if base_cost + self.model.cost_of_singleton(ele) <= self.model.budget:
    #             return False
    #     return True
    #
    # def h(self, n):
    #     if self.is_on_the_edge(n):
    #         return 0
    #     return self.inner_h(n)
    #
    # def f_with_alpha(self, n):
    #     return self.g(n) + self.alpha * self.h(n)
    #
    # def f_without_alpha(self, n):
    #     return self.g(n) + self.h(n)
    #
    # def h_ub0(self, n):
    #     delta, _ = marginal_delta_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
    #                                             budget=self.model.budget - self.model.cost_of_set(n))
    #     return delta
    #
    # def h_ub2(self, n):
    #     delta, _ = marginal_delta_version7_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
    #                                                      budget=self.model.budget - self.model.cost_of_set(n))
    #     return delta
    #
    # def h_dom(self, n):
    #     delta, _ = marginal_delta_dom_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
    #                                                 budget=self.model.budget - self.model.cost_of_set(n))
    #     return delta
    #
    # def lbd0(self, base, budget):
    #     delta, _ = marginal_delta_random_budget(set(base), set(self.model.ground_set) - set(base), self.model,
    #                                             budget=budget)
    #     return delta
    #
    # def lbd2(self, base, budget):
    #     delta, _ = marginal_delta_version7_random_budget(set(base), set(self.model.ground_set) - set(base), self.model,
    #                                                      budget=budget)
    #     return delta
    #
    # def lbd_dom(self, base, budget):
    #     delta, _ = marginal_delta_dom_random_budget(set(base), set(self.model.ground_set) - set(base), self.model,
    #                                                 budget=budget)
    #     return delta
    #
    # def h_ub4(self, n):
    #     opt = DominantOptimizer()
    #     opt.setModel(self.model)
    #     opt.setBase(n)
    #     opt.build()
    #     delta = opt.optimize()['delta']
    #
    #     return delta

    def push_heap(self, s, lbd_v, visited=False, candidate=None, w=None, s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = HeapObj(s, candidate=candidate, w=w, max_idx=max_idx, visited=visited)
        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        v = None
        if final_v >= s_max_v:
            if self.use_alpha:
                lbd_v = min(new_g + self.alpha * new_h, lbd_v)
                v = AugmentedFSValue(new_g + self.alpha * new_h, lbd_v, self.d(s))
            else:
                lbd_v = min(new_g + new_h, lbd_v)
                v = AugmentedFSValue(new_g + new_h, lbd_v, self.d(s))

            node.v = v
            self.max_heap.push(node)

            return node

        return None

    def push_root(self):
        root = HeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        v = AugmentedFSValue(self.f(root), self.f(root), self.d(root.s))
        root.v = v
        root.cost = 0

        f_upper = self.f(root)
        s_max, f_local = self.greedy_add(root.s)
        f_upper = min(f_upper, f_local)

        self.max_heap.push(root)

        return root, f_upper, s_max

    def greedy_add(self, base):
        sol = set(base)
        base_cost = self.model.cost_of_set(list(sol))
        remaining_elements = set(self.model.ground_set) - set(base)
        cur_cost = self.model.cost_of_set(list(sol))

        f_local = None
        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds
            assert u is not None
            if cur_cost + self.model.cost_of_singleton(u) <= self.model.budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(self.model.ground_set) - set(sol),
                                            budget=self.model.budget - base_cost)
            if f_local is None or f_temp < f_local:
                f_local = f_temp

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > self.model.budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        # find the maximum singleton
        v_star, v_star_fv = None, float('-inf')
        for e in set(self.model.ground_set) - set(base):
            if self.model.cost_of_singleton(e) > self.model.budget - self.model.cost_of_set(list(base)):
                # filter out singleton whose cost is larger than budget
                continue
            fv = self.model.objective(list(set(base) | {e}))
            if fv > v_star_fv:
                v_star, v_star_fv = e, fv

        sol_fv = self.model.objective(list(sol))

        if v_star_fv > sol_fv:
            return list(set(base) | {v_star}), f_local
        else:
            return list(sol), f_local

    def optimize(self):
        start_time = time.time()
        # set root node as visited
        root, f_upper, s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = s_max
            stop_time = time.time()

            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}

            return ret

        push_back_count = 0
        s_max_v = self.g(s_max)
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            t0 = time.time()

            node = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v
            max_idx = node.max_idx
            f_upper = min(f_upper, v.lbd_v)

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            if not node.visited:
                s_final, f_local = self.greedy_add(s)
                if self.g(s_final) > self.g(s_max):
                    s_max = s_final
                    s_max_v = self.g(s_max)

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            if self.use_alpha:
                if self.g(s_max) >= f_upper:
                    sol = s_max
                    break
            else:
                if self.g(s_max) >= self.alpha * f_upper:
                    # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
                    sol = s_max
                    break

            if not node.visited and self.pushing_back:
                if f_local < v.lbd_v:
                    push_back_count += 1
                    self.push_heap(s, f_local, True)
                    continue

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            # change this to [max_idx + 1, n]
            # save the cost of s, the g value of s, and the h value of s as properties of its node
            for i in range(max_idx + 1, self.ground_size):
                children_count += 1
                if self.model.cost_of_set(node.s) + self.model.cost_of_singleton(i) <= self.model.budget:
                    self.push_heap(list(set(s) | {i}), node.v.lbd_v, candidate=set(node.candidate) - {i},
                                   w=node.budget - self.model.cost_of_singleton(i))

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}

        return ret


class EfficientBranchAndBound(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.lb_star = None
        self.s_star = None
        self.lbd = None
        self.node_count = 0
        self.basic_mode = False
        self.get_children = None

        self.children_count = 0
        self.time_for_stage_0 = 0
        self.time_for_stage_1 = 0

    def set_h(self, heuristic):
        if heuristic == 'ub0':
            self.lbd = self.lbd0
        elif heuristic == 'ub2':
            self.lbd = self.lbd2
        elif heuristic == 'dom':
            self.lbd = self.lbd_dom

    def build(self):
        self.lb_star = 0
        self.s_star = []
        self.node_count = 0
        if self.basic_mode:
            self.get_children = self.get_children_basic
        else:
            self.get_children = self.get_children_advance

    def greedy_add(self, t):
        base = t.s
        sol = set(base)
        base_cost = self.model.cost_of_set(list(sol))
        remaining_elements = set(t.candidate)
        cur_cost = self.model.cost_of_set(list(sol))

        f_local = self.g(sol) + self.lbd(base=sol, candidate=set(t.candidate) - set(sol),
                                         budget=self.model.budget - base_cost)
        c = []
        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds
            assert u is not None
            if cur_cost + self.model.cost_of_singleton(u) <= self.model.budget:
                # satisfy the knapsack constraint
                sol.add(u)
                c.append(u)
                cur_cost += self.model.cost_of_singleton(u)

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > self.model.budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

            f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(t.candidate) - set(sol),
                                            budget=self.model.budget - base_cost)
            if f_local is None or f_temp < f_local:
                f_local = f_temp

        return list(sol), f_local, c

    def get_children_basic(self, t: BranchAndBoundNode, c):
        children = []
        s = t.s
        tc = list(t.cost)

        tc.sort(key=lambda x: self.g_over([x], s) / self.model.cost_of_singleton(x), reverse=True)

        for i in range(0, len(tc)):
            temp = BranchAndBoundNode(list(set(s) | {tc[i]}), list(set(t.cost) - set(tc[:i + 1])),
                                      t.budget - self.model.cost_of_singleton(tc[i]))
            children.append(temp)

        return children

    def get_children_advance(self, t: BranchAndBoundNode, c):
        children = []
        s = t.s
        for i in range(0, len(c)):
            temp = BranchAndBoundNode(list(set(s) | set(c[:i])), list(set(t.candidate) - set(c[:i + 1])),
                                      t.budget - self.model.cost_of_set(c[:i]))

            if self.model.objective(list(set(s) | set(c[:i]))) + self.lbd0(list(set(s) | set(c[:i])),
                                                                           list(set(t.candidate) - set(c[:i + 1])),
                                                                           t.budget - self.model.cost_of_set(
                                                                               c[:i])) > self.lb_star:
                children.append(temp)

        temp = BranchAndBoundNode(list(set(s) | set(c)), list(set(t.candidate) - set(c)),
                                  t.budget - self.model.cost_of_set(c))

        if self.lbd0(list(set(s) | set(c)), list(set(t.candidate) - set(c)),
                     t.budget - self.model.cost_of_set(c)) > self.lb_star:
            children.append(temp)

        return children

    def bab(self, t: BranchAndBoundNode):
        t0 = time.time()
        self.node_count = self.node_count + 1

        if len(t.candidate) == 0:
            return

        if self.is_on_the_edge(t):
            return

        s_primal, f_local, c = self.greedy_add(t)

        if self.g(s_primal) > self.lb_star:
            self.lb_star = self.g(s_primal)
            self.s_star = s_primal

        ub = f_local
        if self.alpha * ub <= self.lb_star:
            return

        t1 = time.time()

        children = self.get_children(t, c)

        t2 = time.time()

        for t_i in children:
            self.bab(t_i)

        self.time_for_stage_0 += t1 - t0
        self.time_for_stage_1 += t2 - t1
        self.children_count += len(children)

    def optimize(self):
        start_time = time.time()
        self.bab(BranchAndBoundNode(self.s_star, self.model.ground_set, self.model.budget))
        stop_time = time.time()

        ret = {
            'S': self.s_star,
            'c(S)': self.model.cost_of_set(self.s_star),
            'f(S)': self.model.objective(self.s_star),
            'time': stop_time - start_time,
            'node_count': self.node_count,
            'stg0': self.time_for_stage_0,
            'stg1': self.time_for_stage_1,
            'children_count': self.children_count
        }

        return ret


class EfficientBFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0
        self.heap_class = 'tradition'

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def push_heap(self, s, lbd_v, visited=False, first_child=False, heuristic_sequence=None, candidate=None, w=None,
                  s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited, first_child=first_child,
                                   heuristic_sequence=heuristic_sequence, max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        v = None
        if final_v >= s_max_v:
            if self.use_alpha:
                lbd_v = min(new_g + self.alpha * new_h, lbd_v)
                v = RefinedBFSValue(new_g + self.alpha * new_h, lbd_v, self.d(s))
            else:
                lbd_v = min(new_g + new_h, lbd_v)
                v = RefinedBFSValue(new_g + new_h, lbd_v, self.d(s))

            node.v = v

            self.max_heap.push(node)

            return node

        return None

    def greedy_add(self, node: EfficientBFSHeapObj):
        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0

        f_local = None
        heuristic_sequence = []
        # print(f"//")
        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                heuristic_sequence.append(u)
                cur_cost += self.model.cost_of_singleton(u)

            f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(node.candidate) - set(sol), budget=budget)
            if f_local is None or f_temp < f_local:
                f_local = f_temp
                # print(f"base:{base}, sol:{sol}, lbd:{f_temp}, c:{len(candidate)}, budget:{budget}")

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol), f_local, heuristic_sequence

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0

        f_upper = self.f(root)
        s_max, f_local, heuristic_sequence = self.greedy_add(root)
        # f_upper = min(f_upper, f_local)

        v = RefinedBFSValue(self.f(root), min(f_local, f_upper), self.d(root.s))
        root.v = v
        root.heuristic_sequence = heuristic_sequence

        self.max_heap.push(root)

        return root, f_upper, heuristic_sequence, s_max

    def optimize(self):
        start_time = time.time()

        root, f_upper, heuristic_sequence, s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = s_max
            stop_time = time.time()
            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}

            return ret

        push_back_count = 0
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            t0 = time.time()

            node: EfficientBFSHeapObj = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            f_local, heuristic_sequence = node.v.lbd_v, None
            if not node.visited and not node.first_child:
                s_final, f_local, heuristic_sequence = self.greedy_add(node)
                # node.v.lbd_v = min(node.v.lbd_v, f_local)
                f_upper = min(f_upper, node.v.lbd_v)

                if self.g(s_final) > self.g(s_max):
                    s_max = s_final

                if self.use_alpha:
                    if self.g(s_max) >= f_upper:
                        sol = s_max
                        break
                else:
                    if self.g(s_max) >= self.alpha * f_upper:
                        # print(f"here, s:{s_max}, g:{self.g(s_max)}, f:{f_upper}")
                        sol = s_max
                        break

            if node.visited or node.first_child:
                heuristic_sequence = node.heuristic_sequence

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            if not node.visited and self.pushing_back:
                if f_local < v.lbd_v:
                    push_back_count += 1
                    node.v.lbd_v = f_local
                    self.max_heap.push(node)
                    continue

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            # print(f"vis:{node.visited}, f:{node.first_child}, s:{node.s}, is_on_the_edge:{self.is_on_the_edge(node)}, hs:{heuristic_sequence}, c:{len(node.candidate)}, c:{self.model.cost_of_set(node.s)}, w:{node.budget}")

            if self.is_on_the_edge(node):
                continue

            # for i in node.candidate:
            #     if self.model.cost_of_singleton(i) + self.model.cost_of_set(node.s) <= node.budget:
            #         print(f"i:{i}")

            # push first child
            first_ele = heuristic_sequence[0]
            new_candidate = list(set(node.candidate) - {first_ele})
            new_lbd = min(node.v.lbd_v, f_local)
            if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
                open_list_count += 1

                new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
                new_heuristic_sequence.pop(0)

                self.push_heap(s=list(set(s) | {first_ele}), lbd_v=new_lbd, first_child=True,
                               heuristic_sequence=new_heuristic_sequence,
                               candidate=new_candidate,
                               w=node.budget - self.model.cost_of_singleton(first_ele))

            # push second child
            self.push_heap(s=s, lbd_v=new_lbd, first_child=False,
                           candidate=new_candidate, w=node.budget)
            open_list_count += 1

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}

        return ret


class BFSTC(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None

        self.f = self.f_with_alpha
        self.h = None

        self.heap_class = 'simple'

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.h = self.inner_h

    def greedy_add(self, s):
        base = s
        candidate = set(self.model.ground_set) - set(base)
        budget = self.model.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0

        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol)

    def optimize(self):
        start_time = time.time()

        root = BaseHeapObj([], candidate=self.model.ground_set, budget=self.model.budget)
        root.v = self.f(root)

        s_max = self.greedy_add([])
        g_upper = self.alpha * self.h(root)
        self.max_heap.push(root)

        sol = s_max
        node_count = 0
        open_list_count = 0
        while self.max_heap.size() > 0:
            node: BaseHeapObj = self.max_heap.pop()
            node_count += 1

            if self.h(node) == 0:
                sol = node.s
                break

            g_upper = min(g_upper, self.f(node) / self.alpha)

            for i in node.candidate:
                if self.model.cost_of_singleton(i) <= node.budget:
                    s_final = self.greedy_add(set(node.s) | {i})
                    if self.g(s_max) < self.g(s_final):
                        s_max = s_final

                    if self.g(s_max) / g_upper >= self.alpha:
                        sol = s_max
                        break

                    new_node = BaseHeapObj(set(node.s) | {i}, candidate=set(node.candidate) - {i}, budget=node.budget - self.model.cost_of_singleton(i))
                    new_node.v = self.f(new_node)

                    self.max_heap.push(new_node)
                    open_list_count += 1

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count}

        return ret


class EfficientBFSNoInherit(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0
        self.heap_class = 'tradition'

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def push_heap(self, s, lbd_v, visited=False, first_child=False, heuristic_sequence=None, candidate=None, w=None,
                  s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = EfficientBFSHeapObj(s, candidate=candidate, w=w, visited=visited, first_child=first_child,
                                   heuristic_sequence=heuristic_sequence, max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        if final_v >= s_max_v:
            if self.use_alpha:
                node.v = new_g + self.alpha * new_h
            else:
                node.v = new_g + new_h

            self.max_heap.push(node)

            return node

        return None

    def greedy_add(self, node: EfficientBFSHeapObj):
        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0

        # f_local = None
        heuristic_sequence = []
        # print(f"//")
        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                heuristic_sequence.append(u)
                cur_cost += self.model.cost_of_singleton(u)

            # f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(node.candidate) - set(sol), budget=budget)
            # if f_local is None or f_temp < f_local:
            #     f_local = f_temp
            #     print(f"base:{base}, sol:{sol}, lbd:{f_temp}, c:{len(candidate)}, budget:{budget}")

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol), heuristic_sequence

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0

        f_upper = self.f(root)
        s_max, heuristic_sequence = self.greedy_add(root)

        root.v = f_upper
        root.heuristic_sequence = heuristic_sequence

        self.max_heap.push(root)

        return root, f_upper, heuristic_sequence, s_max

    def optimize(self):
        start_time = time.time()

        root, f_upper, heuristic_sequence, s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = s_max
            stop_time = time.time()
            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}

            return ret

        push_back_count = 0
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            t0 = time.time()

            node: EfficientBFSHeapObj = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            f_local, heuristic_sequence = 0., None
            if not node.visited and not node.first_child:
                s_final, heuristic_sequence = self.greedy_add(node)
                # node.v.lbd_v = min(node.v.lbd_v, f_local)
                f_upper = min(f_upper, v)

                if self.g(s_final) > self.g(s_max):
                    s_max = s_final

                if self.use_alpha:
                    if self.g(s_max) >= f_upper:
                        sol = s_max
                        break
                else:
                    if self.g(s_max) >= self.alpha * f_upper:
                        # print(f"here, s:{s_max}, g:{self.g(s_max)}, f:{f_upper}")
                        sol = s_max
                        break

            if node.visited or node.first_child:
                heuristic_sequence = node.heuristic_sequence

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            if not node.visited and self.pushing_back:
                if f_local < v:
                    push_back_count += 1
                    node.v.lbd_v = f_local
                    self.max_heap.push(node)
                    continue

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            # print(f"vis:{node.visited}, f:{node.first_child}, s:{node.s}, is_on_the_edge:{self.is_on_the_edge(node)}, hs:{heuristic_sequence}, c:{len(node.candidate)}, c:{self.model.cost_of_set(node.s)}, w:{node.budget}")

            if self.is_on_the_edge(node):
                continue

            # for i in node.candidate:
            #     if self.model.cost_of_singleton(i) + self.model.cost_of_set(node.s) <= node.budget:
            #         print(f"i:{i}")

            # push first child
            first_ele = heuristic_sequence[0]
            new_candidate = list(set(node.candidate) - {first_ele})
            if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
                open_list_count += 1

                new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
                new_heuristic_sequence.pop(0)

                self.push_heap(s=list(set(s) | {first_ele}), lbd_v=v, first_child=True,
                               heuristic_sequence=new_heuristic_sequence,
                               candidate=new_candidate,
                               w=node.budget - self.model.cost_of_singleton(first_ele))

            # push second child
            self.push_heap(s=s, lbd_v=v, first_child=False,
                           candidate=new_candidate, w=node.budget)
            open_list_count += 1

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}

        return ret


class InheritBFS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)
        self.max_heap = None
        self.inner_h = None
        self.f = None
        self.d = None
        self.lbd = None
        self.use_alpha = False
        self.pushing_back = True
        self.ground_size = 0
        self.heap_class = 'tradition'

    def build(self):
        if self.heap_class == 'tradition':
            self.max_heap = MaxHeap()
        elif self.heap_class == 'simple':
            self.max_heap = SimpleMaxHeap()

        self.max_heap.clear()
        self.ground_size = len(self.model.ground_set)

        if self.use_alpha:
            self.f = self.f_with_alpha
        else:
            self.f = self.f_without_alpha

    def push_heap(self, s, lbd_v, visited=False, candidate=None, w=None,
                  s_max_v=0):
        max_idx = 0
        if len(s) > 0:
            max_idx = max(s)

        node = HeapObj(s, candidate=candidate, w=w, visited=visited, max_idx=max_idx)
        node.cost = self.model.cost_of_set(s)

        new_g = self.g(node)
        new_h = self.h(node)
        final_v = new_g + new_h

        v = None
        if final_v >= s_max_v:
            if self.use_alpha:
                lbd_v = min(new_g + self.alpha * new_h, lbd_v)
                v = RefinedBFSValue(new_g + self.alpha * new_h, lbd_v, self.d(s))
            else:
                lbd_v = min(new_g + new_h, lbd_v)
                v = RefinedBFSValue(new_g + new_h, lbd_v, self.d(s))

            node.v = v

            self.max_heap.push(node)

            return node

        return None

    def greedy_add(self, node: EfficientBFSHeapObj):
        base = node.s
        candidate = node.candidate
        budget = node.budget

        sol = set(base)
        remaining_elements = set(candidate)
        cur_cost = 0

        f_local = None
        # print(f"//")
        while len(remaining_elements):
            u, max_density = None, -1.
            for e in remaining_elements:
                # e is an object
                ds = self.model.density(e, list(sol))
                if u is None or ds > max_density:
                    u, max_density = e, ds

            assert u is not None

            if cur_cost + self.model.cost_of_singleton(u) <= budget:
                # satisfy the knapsack constraint
                sol.add(u)
                cur_cost += self.model.cost_of_singleton(u)

            f_temp = self.g(sol) + self.lbd(base=sol, candidate=set(node.candidate) - set(sol), budget=budget)
            if f_local is None or f_temp < f_local:
                f_local = f_temp
                # print(f"base:{base}, sol:{sol}, lbd:{f_temp}, c:{len(candidate)}, budget:{budget}")

            remaining_elements.remove(u)
            # filter out violating elements
            to_remove = set()
            for v in remaining_elements:
                if self.model.cost_of_singleton(v) + cur_cost > budget:
                    to_remove.add(v)
            remaining_elements -= to_remove

        return list(sol), f_local

    def push_root(self):
        root = EfficientBFSHeapObj([], candidate=self.model.ground_set, w=self.model.budget, visited=True, max_idx=0)
        root.cost = 0

        f_upper = self.f(root)
        s_max, f_local = self.greedy_add(root)
        # f_upper = min(f_upper, f_local)

        v = RefinedBFSValue(self.f(root), min(f_upper, f_local), self.d(root.s))
        root.v = v

        self.max_heap.push(root)

        return root, f_upper, s_max

    def optimize(self):
        start_time = time.time()

        root, f_upper, s_max = self.push_root()

        # check if s_max now is an optimal solution
        if self.g(s_max) >= self.alpha * f_upper:
            # print(f"here, g:{self.g(s_max)}, f:{f_upper}")
            sol = s_max
            stop_time = time.time()
            ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
                   'time': stop_time - start_time, 'node_count': 1, "open_list_count": 1,
                   "push_back_count": 0}

            return ret

        push_back_count = 0
        sol = s_max

        node_count = 0
        open_list_count = 1

        time_for_stage_0 = 0
        time_for_stage_1 = 0
        time_for_stage_2 = 0
        time_for_stage_3 = 0
        children_count = 0

        while self.max_heap.size() > 0:
            t0 = time.time()

            node: EfficientBFSHeapObj = self.max_heap.pop()
            node_count += 1
            s = node.s
            v = node.v

            t1 = time.time()

            time_for_stage_0 += t1 - t0

            f_local = node.v.lbd_v
            if not node.visited:
                s_final, f_local = self.greedy_add(node)
                # node.v.lbd_v = min(node.v.lbd_v, f_local)
                f_upper = min(f_upper, node.v.lbd_v)

                if self.g(s_final) > self.g(s_max):
                    s_max = s_final

                if self.use_alpha:
                    if self.g(s_max) >= f_upper:
                        sol = s_max
                        break
                else:
                    if self.g(s_max) >= self.alpha * f_upper:
                        # print(f"here, s:{s_max}, g:{self.g(s_max)}, f:{f_upper}")
                        sol = s_max
                        break

            t2 = time.time()

            time_for_stage_1 += t2 - t1

            t3 = time.time()

            time_for_stage_2 += t3 - t2

            # print(f"vis:{node.visited}, f:{node.first_child}, s:{node.s}, is_on_the_edge:{self.is_on_the_edge(node)}, hs:{heuristic_sequence}, c:{len(node.candidate)}, c:{self.model.cost_of_set(node.s)}, w:{node.budget}")

            if self.is_on_the_edge(node):
                continue

            # for i in node.candidate:
            #     if self.model.cost_of_singleton(i) + self.model.cost_of_set(node.s) <= node.budget:
            #         print(f"i:{i}")

            # # push first child
            # first_ele = heuristic_sequence[0]
            # new_candidate = list(set(node.candidate) - {first_ele})
            # if node.cost + self.model.cost_of_singleton(first_ele) <= self.model.budget:
            #     open_list_count += 1
            #
            #     new_heuristic_sequence = copy.deepcopy(heuristic_sequence)
            #     new_heuristic_sequence.pop(0)
            #
            #     self.push_heap(s=list(set(s) | {first_ele}), lbd_v=node.v.lbd_v, first_child=True,
            #                    heuristic_sequence=new_heuristic_sequence,
            #                    candidate=new_candidate,
            #                    w=node.budget - self.model.cost_of_singleton(first_ele))
            #
            # # push second child
            # self.push_heap(s=s, lbd_v=node.v.lbd_v, first_child=False,
            #                candidate=new_candidate, w=node.budget)
            # open_list_count += 1

            new_lbd = min(f_local, node.v.lbd_v)
            for i in node.candidate:
                if i > node.max_idx and node.cost + self.model.cost_of_singleton(i) <= self.model.budget:
                    open_list_count += 1
                    self.push_heap(s=list(set(s) | {i}), lbd_v=new_lbd, candidate=set(node.candidate) - {i}, w = node.budget- self.model.cost_of_singleton(i))

            t4 = time.time()

            time_for_stage_3 += t4 - t3

        stop_time = time.time()

        assert sol is not None, "No solution found."

        ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
               'time': stop_time - start_time, 'node_count': node_count, "open_list_count": open_list_count,
               "push_back_count": push_back_count, "stg0": time_for_stage_0, "stg1": time_for_stage_1,
               "stg2": time_for_stage_2, "stg3": time_for_stage_3,
               'children_count': children_count}

        return ret
