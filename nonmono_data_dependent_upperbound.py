from base_task import BaseTask

from typing import Set, List
from itertools import accumulate
import bisect
import numpy as np


def singleton_knapsack_fill(model: BaseTask) -> float:
    # sort by density
    eles = sorted([x for x in model.ground_set], key=lambda x : model.density(x, base=[]), reverse=True)
    cumsum_cost = list(accumulate([model.cost_of_singleton(e) for e in eles]))
    idx = bisect.bisect_left(cumsum_cost, model.budget)
    return sum([model.objective([e]) for e in eles[:idx + 1]])
