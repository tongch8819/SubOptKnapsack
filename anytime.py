import time

from MaxHeap import HeapObj, MaxHeap
from data_dependent_upperbound import marginal_delta_random_budget, marginal_delta_m_acc_random_budget, \
    marginal_delta_version7_random_budget, marginal_delta, marginal_delta_m_acc, marginal_delta_version7


# we keep running the algorithm till time limit is reached or we get an optimal solution
# keep all the

def getATOptimizer(alg, model, heuristic, time_limit, eps):
    opt = None
    if alg == 'AFS':
        opt = AnytimeFocalSearch()
    elif alg == 'ARA*':
        opt = ARAstar().eps(eps)

    return opt.set_tl(time_limit).set_model(model).set_eps(eps)


class AnytimeOptimizer:
    def __init__(self):
        self.eps = None
        self.tl = None
        self.model = None

        self.bound = None
        self.w = None

        self.af_list = []
        self.af = None
        self.sol_list = []
        self.sol = None
        self.h = None

        self.f_max = None
        pass

    def set_tl(self, time_limit):
        self.tl = time_limit
        return self

    def set_model(self, model):
        self.model = model
        return self

    def set_heuristic(self, heuristic):
        if heuristic == 'ub0':
            self.h = self.h_ub0
        elif heuristic == 'ub1':
            self.h = self.h_ub1
        elif heuristic == 'ub2':
            self.h = self.h_ub2

        return self

    def g(self, s):
        return self.model.objective(list(s))

    def f(self, s):
        pass

    def set_eps(self, eps):
        self.eps = eps
        return self

    # initialize before iterations
    def init(self):
        pass

    # the heuristic function
    def h_ub0(self, S):
        delta, _ = marginal_delta(set(S), set(self.model.ground_set) - set(S), self.model)
        # delta, _ = marginal_delta_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub1(self, S):
        delta, _ = marginal_delta_m_acc(set(S), set(self.model.ground_set) - set(S), self.model)
        # delta, _ = marginal_delta_m_acc_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # the heuristic function
    def h_ub2(self, S):
        delta, _ = marginal_delta_version7(set(S), set(self.model.ground_set) - set(S), self.model)
        # delta, _ = marginal_delta_version7_random_budget(set(S), set(self.model.ground_set) - set(S), self.model, budget=self.model.budget - self.model.cost_of_set(S))
        return delta

    # find the next bound with current solution
    def get_next_bound(self):
        self.bound = self.model.objective(self.sol) + self.eps
        pass

    def is_goal(self, s):
        pass

    # given a certain bound, improve the function value of sol to at least this bound
    def improve(self):
        pass

    def report(self):
        self.af = self.model.objective(self.sol)/self.f_max
        self.af_list.append(self.af)
        self.sol_list.append(self.sol)

    def optimize(self):
        start_time = time.time()

        while True:
            valid = self.improve()
            if not valid:
                print(f"No valid solution. Terminate.")
                break

            self.report()

            if self.af >= 1:
                print(f"Found the optimal solution. Terminate.")
                break
            running_time = time.time() - start_time
            if running_time >= self.tl:
                print(f"Time has been used up. Terminate.")
                break

            self.get_next_bound()

        return self.sol_list, self.af_list


class AnytimeFocalSearch(AnytimeOptimizer):
    def __init__(self):
        super().__init__()
        self.open = MaxHeap()
        self.focal = MaxHeap()

    def f(self, s):
        return self.g(s) + self.h(s)

    def init(self):
        self.open.clear()
        self.focal.clear()

    def get_next_bound(self):
        self.bound = self.model.objective(self.sol) + self.eps
        self.w = self.open.top().v/self.bound
        if self.w > 1:
            self.w = 1

    def wrap(self, s):
        return HeapObj(s, self.f(s))

    def add_succ(self, n, f_min):
        cost_c = self.model.cost(n.s)
        for i in set(self.model.ground_set) - set(n.s):
            if self.model.cost_of_singleton(i) + cost_c <= self.model.budget:
                new_s = set(n.s) | {i}
                self.open.push(self.wrap(new_s))
                if self.f(new_s) <= self.bound:
                    self.focal.push(self.wrap(new_s))

    def update_upper_bound(self, old_b, new_b):
        for n in self.open.h:
            if old_b > self.f(n.s) > new_b:
                self.focal.push(n)

    def is_goal(self, n):
        if self.g(n.s) >= self.w * self.open.top().v:
            return True
        return False

    def improve(self):
        n_start = self.wrap(set())

        self.open.push(n_start)
        self.focal.push(n_start)

        while self.focal.size() > 0:
            f_min = self.open.top().v
            n = self.focal.pop()
            self.open.remove(n)

            if self.is_goal(n):
                self.sol = n.s
                self.af = self.w
                break

            self.add_succ(n, f_min)

            if self.open.size() > 0 and f_min < self.open.top().v:
                self.update_upper_bound(self.w * f_min, self.w * self.open.top().v)


class ARAstar(AnytimeOptimizer):
    def __init__(self):
        super().__init__()

