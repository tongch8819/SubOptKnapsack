import time

from base_task import BaseTask

import numpy as np

# for modular knapsack problems
def dp(model: BaseTask):
    v = list(model.ground_set)
    b = int(model.budget)

    n = len(v)

    dp = np.ndarray(shape=(n + 1, b + 1), dtype=float)

    dp[0][0] = 0

    for i in range(0, n + 1):
        dp[i][0] = 0

    for j in range(0, b + 1):
        dp[0][b] = 0

    for i in range(1, n + 1):
        for j in range(1, b + 1):
            cost = model.cost_of_singleton(v[i-1])
            gain = model.objective([v[i-1]])

            if cost <= j:
                left = dp[i-1][int(j-cost)] + gain
            else:
                left = 0

            right = dp[i-1][j]

            dp[i][j] = max(left, right)

    res = {
        "f(S)": dp[n][b]
    }

    return res