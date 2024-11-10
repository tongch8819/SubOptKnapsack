import random
import numpy as np

from base_task import BaseTask

def estimate_marginal_gain(model: BaseTask, x, e, s):
    samples = []
    for i in range(0, s):
        S = set()
        for ele in x.keys():
            p = x[ele]
            if random.random() <= p:
                S.add(ele)
        samples.append(model.marginal_gain(e, S))

    return np.mean(samples)