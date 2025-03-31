import argparse
import os
import pickle
import random

import numpy as np

import model_factory
from base_task import BaseTask
from mgreedy import modified_greedy_plain


# original greedy value has computed
# the random value
# the new value

class SolutionProcessor:
    def __init__(self):
        self.task = None
        self.model = None
        self.n = 0
        self.c = None

        self.bds = None

        self.seed = None

    def set_model(self, model):
        self.model = model

    def set_seed(self, seed):
        self.seed = seed

    def set_b(self, start_point=6, num_points=35, interval=1):
        end_point = start_point + (num_points - 1) * interval
        self.bds = np.linspace(start=start_point, stop=end_point, num=num_points)

    def set_task(self, task):
        self.task = task

    def build(self):
        self.n = len(self.model.ground_set)
        self.c = np.zeros(self.n)
        for i in range(0, self.n):
            self.c[i] = self.model.cost_of_singleton(i)

        np.random.seed(self.seed)
        random.seed(self.seed)

    def get_random(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

        results = {}

        for b in self.bds:
            x = []
            cost = 0
            for i in range(0, self.n):
                if cost + self.c[i] <= b:
                    if random.random() <= 0.5:
                        x.append(i)
                        cost += self.c[i]

            results[float(b)] = self.model.objective(x)

        return results

    def get_greedy(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

        result = {}

        for b in self.bds:
            self.model.budget = b
            res = modified_greedy_plain(self.model)
            result[float(b)] = res['f(S)']
            print(f"b:{b}, f:{res['f(S)']}, S:{res['S']}")

        # archive = './result/archive-5'
        # result = {}
        #
        # source_dir = os.path.join(archive, f"{self.task}", f"{self.n}", f"{self.seed}")
        # for name in os.listdir(source_dir):
        #     if not os.path.isdir(os.path.join(source_dir, name)):
        #         _, up, task, budget = name.strip()[:-5].split('-')
        #         if up == 'ub1':
        #             file_path = os.path.join(source_dir, name)
        #             with open(file_path, "rb") as rd:
        #                 kv_data = pickle.load(rd)
        #                 result[float(budget)] = kv_data['f(S)']
        #                 print(f"budget:{budget}, c:{kv_data['c(S)']}, s:{kv_data['S']}")
        return result

    def get_upb0(self):
        ground = list(self.model.ground_set)
        ground.sort(key=lambda y: self.model.density(y, []), reverse=True)

        result = {}

        for b in self.bds:
            x = []
            cost = 0
            for i in range(0, self.n):
                if cost + self.c[ground[i]] <= b:
                    x.append(ground[i])
                    cost += self.c[ground[i]]

            print(f"b:{b}, cost:{cost}, s:{x}, f:{self.model.objective({x[0]})}")
            result[float(b)] = self.model.objective(x)

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", '--task', help="task na"
                                             "me")
    parser.add_argument("-n", '--num', type=int, help="ground set size")
    parser.add_argument("-ss", '--stseed', default=100, type=int, help="start seed")
    parser.add_argument("-sp", '--spseed', default=200, type=int, help="stop seed")

    args = parser.parse_args()

    start_seed = args.stseed
    stop_seed = args.spseed

    for seed in range(start_seed, stop_seed):
        save_dir = os.path.join('./result', 'archive-34', f"{args.task}", f"{args.num}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        model = model_factory.model_factory(args.task, args.num, seed, 0, True, cm='normal')

        processor = SolutionProcessor()
        processor.set_task(args.task)
        processor.set_model(model)
        processor.set_seed(seed)
        processor.set_b()
        processor.build()

        # s_r = processor.get_random()
        s_g = processor.get_greedy()
        # s_u = processor.get_upb0()

        # save_path = os.path.join(save_dir, "{}-{}-{}.pckl".format(
        #     args.task, 'random', seed))
        #
        # with open(save_path, "wb") as wrt:
        #     pickle.dump(s_r, wrt)
        # print(f"seed:{seed}, random completed.")
        # print(s_r)

        save_path = os.path.join(save_dir, "{}-{}-{}.pckl".format(
            args.task, 'greedy', seed))

        with open(save_path, "wb") as wrt:
            pickle.dump(s_g, wrt)
        print(f"seed:{seed}, greedy completed.")
        print(s_g)

        # save_path = os.path.join(save_dir, "{}-{}-{}.pckl".format(
        #     args.task, 'upper', seed))
        #
        # with open(save_path, "wb") as wrt:
        #     pickle.dump(s_u, wrt)
        # print(f"seed:{seed}, upper completed.")
        # print(s_u)
