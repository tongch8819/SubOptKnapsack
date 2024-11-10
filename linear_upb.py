import os
import pickle
import time

import mgreedy
import optimizer
from compute_knapsack_exp import model_factory


def test():
    n = 500
    task = "facebook"
    root_dir = os.path.join("./result", "archive-10")
    if not os.path.exists(os.path.join(root_dir, task, f"{n}")):
        os.mkdir(os.path.join(root_dir, task, f"{n}"))

    for seed in range(0, 1):

        for budget in range(6, 7):
            start = time.time()

            model = model_factory("facebook", 500, seed, budget, cm="normal", knap=True)

            opt: optimizer.Optimizer = optimizer.Optimizer().set_model(model)

            # opt.permutation_max()

            opt.permutation_random(seed = seed)

            res_opt = opt.optimize()

            stop = time.time()

            start2 = time.time()

            res = mgreedy.modified_greedy_ub7(model)  # dict

            stop2 = time.time()

            print(f"res opt x:{res_opt['x']}")

            x = res_opt['x']

            total = 0
            for i in x.keys():
                print(opt.diag_s[i])
                total += model.objective([i]) * x[i] * opt.diag_s[i]
            print(total)

            final_res = {
                "f(S)": res['f(S)'],
                "ub1": res["Lambda"],
                "upb_opt": float(res_opt['upb']),
                "AF_opt": float(res['f(S)']/res_opt['upb']),
                "AF": float(res['f(S)'] / res['Lambda']),
                "RAF": float(res_opt["upb"] / res['Lambda']),
                "time_opt": stop-start,
                "time":stop2 - start2
            }

            print(f"Done:seed:{seed}, budget:{budget}, result:{final_res}")

            save_dir = os.path.join(root_dir, task, f"{n}", f"{seed}")

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_path = os.path.join(save_dir, "{}-{}-{}-{:.2f}.pckl".format(
                "modified_greedy", "ub1mv", model.__class__.__name__, budget))

            with open(save_path, "wb") as wrt:
                pickle.dump(final_res, wrt)


    pass


if __name__ == "__main__":
    root_dir = "./result"
    test()
