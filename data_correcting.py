from typing import Set

from OptimalAlg import OptimalAlg
from base_task import BaseTask


class DCA(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

        self.eps = None
        self.count = 0

    def f(self, x):
        x = list(x)
        return self.model.objective(x)

    def c(self, x):
        if type(x) == int:
            return self.model.cost_of_singleton(x)
        x = list(x)
        return self.model.cost_of_set(x)

    def delta_minus(self, phi: Set[int], I: Set[int], i):
        return self.f(phi) - self.f(phi | {i})

    def delta_plus(self, phi: Set[int], I: Set[int], i):
        return self.f(I) - self.f(I - {i})

    def build(self):
        self.count = 0

        if self.eps is None:
            total = self.f(self.model.ground_set)
            # print(f"alpha:{self.alpha}, total:{total}")
            self.eps = (1 - self.alpha) * total
            # print(f"eps:{self.eps}")

    # step 1
    def check_equal(self, phi: Set[int], I: Set[int]):
        if phi == I:
            return True
        else:
            interval = I - phi
            cost_start = self.c(phi)
            remaining_budget = self.model.budget - cost_start
            for i in interval:
                if self.c(i) <= remaining_budget:
                    return False

            return True

    # step 2
    def check_left(self, phi: Set[int], I: Set[int]):
        interval = I - phi
        max_delta_minus = None

        for i in interval:
            if max_delta_minus is None or self.delta_minus(phi, I, i) > max_delta_minus:
                max_delta_minus = self.delta_minus(phi, I, i)

        if max_delta_minus >= 0:
            r_minus = None
            for i in interval:
                if self.delta_minus(phi, I, i) == max_delta_minus:
                    r_minus = i
                    break
            return True, r_minus
        else:
            return False, None

    # step 3
    def check_right(self, phi: Set[int], I: Set[int]):
        interval = I - phi
        max_delta_plus = None
        cost_start = self.c(phi)
        remaining_budget = self.model.budget - cost_start

        for i in interval:
            if self.c(i) <= remaining_budget:
                if max_delta_plus is None or self.delta_plus(phi, I, i) > max_delta_plus:
                    max_delta_plus = self.delta_plus(phi, I, i)

        if max_delta_plus is not None and max_delta_plus >= 0:
            r_plus = None
            for i in interval:
                if self.c(i) <= remaining_budget:
                    if self.delta_plus(phi, I, i) == max_delta_plus:
                        r_plus = i
                        break
            return True, r_plus
        else:
            return False, None

    # step 4
    def check_left_with_eps(self, phi: Set[int], I: Set[int], eps: float):
        interval = I - phi
        max_delta_minus = None
        for i in interval:
            if max_delta_minus is None or self.delta_minus(phi, I, i) > max_delta_minus:
                max_delta_minus = self.delta_minus(phi, I, i)

        if max_delta_minus >= eps:
            r_minus = None

            for i in interval:
                if self.delta_plus(phi, I, i) == max_delta_minus:
                    r_minus = i
                    break
            return True, r_minus, max_delta_minus
        else:
            return False, None, 0

    # step 5
    def check_right_with_eps(self, phi: Set[int], I: Set[int], eps: float):
        interval = I - phi
        max_delta_plus = None
        cost_start = self.c(phi)
        remaining_budget = self.model.budget - cost_start

        for i in interval:
            if self.c(i) <= remaining_budget:
                if max_delta_plus is None or self.delta_plus(phi, I, i) > max_delta_plus:
                    max_delta_plus = self.delta_plus(phi, I, i)

        if max_delta_plus is not None and max_delta_plus >= eps:
            r_plus = None
            for i in interval:
                if self.c(i) <= remaining_budget:
                    if self.delta_minus(phi, I, i) == max_delta_plus:
                        r_plus = i
                        break
            return True, r_plus, max_delta_plus
        else:
            return False, None, 0

    # step 6
    def check_branching_rule(self, phi: Set[int], I: Set[int]):
        k, k_value = None, 0
        interval = I - phi
        cost_start = self.c(phi)
        remaining_budget = self.model.budget - cost_start

        for i in interval:
            if self.c(i) <= remaining_budget:
                temp_k_value = min(self.delta_plus(phi, I, i), self.delta_minus(phi, I, i))
                if temp_k_value < k_value:
                    k_value = temp_k_value
                    k = i

        return k

    def DC(self, phi: Set[int], I: Set[int], eps: float):
        # step 1
        if self.check_equal(phi, I):
            return phi, 0
        # step 2
        flag, r_plus = self.check_right(phi, I)
        if flag:
            lbd, gamma = self.DC(phi | {r_plus}, I, eps)
            return lbd, gamma

        # step 3
        flag, r_minus = self.check_left(phi, I)
        if flag:
            lbd, gamma = self.DC(phi, I - {r_minus}, eps)
            return lbd, gamma

        # step 4
        flag, r_plus, dp = self.check_right_with_eps(phi, I, eps)
        if flag:
            lbd, gamma = self.DC(phi | {r_plus}, I, eps - dp)
            gamma += dp
            return lbd, gamma

        # step 5
        flag, r_minus, dp = self.check_left_with_eps(phi, I, eps)
        if flag:
            lbd, gamma = self.DC(phi, I - {r_minus}, eps - dp)
            gamma += dp
            return lbd, gamma

        # step 6
        k = self.check_branching_rule(phi, I)
        lbd_p, gamma_p = self.DC(phi | {k}, I, eps)
        lbd_m, gamma_m = self.DC(phi, I - {k}, eps)
        lbd, gamma = None, 0
        if self.f(lbd_p) > self.f(lbd_m):
            lbd = lbd_p
        else:
            lbd = lbd_m

        gamma = max(self.f(lbd_p), self.f(lbd_m)) - max(self.f(lbd_p) - gamma_p, self.f(lbd_m) - gamma_m)
        # print(f"count:{self.count}")
        # self.count = self.count + 1
        # step 7
        return lbd, gamma

    def optimize(self):
        lbd, gamma = self.DC(set(), set(self.model.ground_set), self.eps)
        return {
            "S": lbd,
            "f(S)": self.f(lbd),
            "gamma": gamma,
            "eps": self.eps
        }
