from base_task import BaseTask

from typing import Set, List
from itertools import accumulate
import bisect
import numpy as np


def singleton_knapsack_fill(model: BaseTask) -> float:
    # sort by density in descending order
    eles = sorted([x for x in model.ground_set], key=lambda x : model.density(x, base=[]), reverse=True)
    cumsum_cost = list(accumulate([model.cost_of_singleton(e) for e in eles]))
    idx = bisect.bisect_left(cumsum_cost, model.budget)
    res = sum([model.objective([e]) for e in eles[:idx]])
    # add fractional solution
    # cur_total_cost = model.cost_of_set(eles[:idx + 1])
    cur_total_cost = cumsum_cost[idx - 1] if idx >= 1 else 0
    fractional_ele = eles[idx]
    assert cur_total_cost <= model.budget, "{} ?< {}, List: {}".format(cur_total_cost, model.budget, cumsum_cost)
    ratio = (model.budget - cur_total_cost) / model.cost_of_singleton(fractional_ele)
    res += model.objective([fractional_ele]) * ratio
    return res
