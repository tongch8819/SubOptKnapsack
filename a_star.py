# import time
# from functools import total_ordering
#
# from OptimalAlg import OptimalAlg
# from base_task import BaseTask
# from MaxHeap import MaxHeap, HeapObj
# from data_dependent_upperbound import marginal_delta_version7, marginal_delta, marginal_delta_random_budget, \
#     marginal_delta_version7_random_budget
#
#
# # class A_star(OptimalAlg):
# #     def __init__(self, model: BaseTask):
# #         super().__init__(model)
# #
# #         self.closed_list = []
# #         self.heap = MaxHeap()
# #         self.h = None
# #         self.alpha = 1
# #
# #         self.E = set()
# #
# #     def build(self):
# #         self.closed_list.clear()
# #         self.heap.clear()
# #         if self.opt == 'ub0':
# #             self.h = self.h_ub0
# #         elif self.opt == 'ub2':
# #             self.h = self.h_ub2
# #
# #     def f(self, s):
# #         return self.g(s) + self.h(s)
# #
# #     # the heuristic function
# #     def h_ub0(self, S):
# #         delta, _ = marginal_delta(set(S), set(self.model.ground_set) - set(S), self.model)
# #         return delta
# #
# #     # the heuristic function
# #     def h_ub2(self, S):
# #         delta, _ = marginal_delta_version7(set(S), set(self.model.ground_set) - set(S), self.model)
# #         return delta
# #
# #     def g(self, S):
# #         return self.f(list(S)) + self.alpha * self.h(S)
# #
# #     def greedy_with_base(self, S):
# #         model = self.model
# #
# #         sol = S
# #         remaining_elements = set(model.ground_set) - S
# #         cur_cost = model.cost_of_set(list(S))
# #
# #         while len(remaining_elements):
# #             u, max_density = None, -1.
# #             for e in remaining_elements:
# #                 # e is an object
# #                 ds = self.model.density(e, list(sol))
# #                 if u is None or ds > max_density:
# #                     u, max_density = e, ds
# #             assert u is not None
# #             if cur_cost + model.cost_of_singleton(u) <= model.budget:
# #                 # satisfy the knapsack constraint
# #                 sol.add(u)
# #                 cur_cost += model.cost_of_singleton(u)
# #
# #             remaining_elements.remove(u)
# #             # filter out violating elements
# #             to_remove = set()
# #             for v in remaining_elements:
# #                 if model.cost_of_singleton(v) + cur_cost > model.budget:
# #                     to_remove.add(v)
# #             remaining_elements -= to_remove
# #
# #         return sol
# #
# #     def optimize(self):
# #         start_time = time.time()
# #
# #         ret = {
# #
# #         }
# #
# #         s_star = set()
# #         s_star_v = self.model.objective(list(s_star))
# #
# #         L = MaxHeap()
# #         L.push(HeapObj(s_star, self.f(s_star)))
# #
# #         while L.size() > 0:
# #             obj = L.pop()
# #             s, f_s = obj.s, obj.v
# #             if f_s > s_star_v:
# #                 s_plus = self.greedy_with_base(s)
# #                 if self.model.objective(s_plus) > s_star_v:
# #                     s_star_v = self.model.objective(s_plus)
# #                     s_star = s_plus
# #                 for ele in set(self.model.ground_set) - s:
# #                     t = s | {ele}
# #                     if self.model.cost_of_set(t) <= self.model.budget and self.g(t) >= s_star_v:
# #                         L.push(HeapObj(t, self.g(t)))
# #
# #         stop_time = time.time()
# #
# #         ret['S'] = s_star
# #         ret['c(S)'] = self.model.cost_of_set(s_star)
# #         ret['f(S)'] = s_star_v
# #         ret['time'] = stop_time - start_time
# #
# #         return ret
#
# @total_ordering
# class AstarValue:
#     def __init__(self, value, max_idx):
#         self.value = value
#         self.max_idx = max_idx
#
#     def __eq__(self, other):
#         return self.value == other.value and self.value == other.value
#
#     def __lt__(self, other):
#         if self.value < other.value:
#             return True
#         return False
#
#
# class Astar(OptimalAlg):
#     def __init__(self, model):
#         super().__init__(model)
#         self.max_heap = MaxHeap()
#         self.h = None
#
#     def build(self):
#         self.max_heap.clear()
#
#     def g(self, n):
#         return self.model.objective(list(n))
#
#     def h_ub0(self, n):
#         delta, _ = marginal_delta_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
#                                                 budget=self.model.budget - self.model.cost_of_set(n))
#         return delta
#
#     def h_ub2(self, n):
#         delta, _ = marginal_delta_version7_random_budget(set(n), set(self.model.ground_set) - set(n), self.model,
#                                                          budget=self.model.budget - self.model.cost_of_set(n))
#         return delta
#
#     def f(self, n):
#         return self.g(n) + self.h(n)
#
#     def push_heap(self, n):
#         max_value = 0
#         if len(n) > 0:
#             max_value = max(n)
#
#         v = AstarValue(self.f(n), max_value)
#         node = HeapObj(n, v)
#         self.max_heap.push(node)
#
#     def set_h(self, heuristic):
#         if heuristic == 'ub0':
#             self.h = self.h_ub0
#         elif heuristic == 'ub2':
#             self.h = self.h_ub2
#
#     def is_goal(self, n):
#         if self.g(n.s) >= self.alpha * self.f(n.s):
#             return True
#         return False
#
#     def optimize(self):
#         start_time = time.time()
#
#         root = []
#         self.push_heap(root)
#
#         sol = None
#         node_count = 0
#         while self.max_heap.size() > 0:
#             node = self.max_heap.pop()
#
#             s = node.s
#             v = node.v
#             max_idx = v.max_idx
#
#             node_count+=1
#             if self.is_goal(node):
#                 sol = node.s
#                 break
#
#             for i in set(self.model.ground_set) - set(s):
#                 if i > max_idx and self.model.cost_of_set(s) + self.model.cost_of_singleton(i) <= self.model.budget:
#                     self.push_heap(list(set(s) | {i}))
#
#
#         assert sol is not None, "No solution found."
#
#         stop_time = time.time()
#
#         ret = {'S': sol, 'c(S)': self.model.cost_of_set(sol), 'f(S)': self.model.objective(sol),
#                'time': stop_time - start_time, 'node_count': node_count}
#
#         return ret