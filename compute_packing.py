import argparse
import os
import pickle
import time

from MWU import MWU, greedy_for_matroid
from compute_knapsack_exp import model_factory
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='facebook', help='task to test')
    parser.add_argument("-n",'--num', default=50, help='size of the ground set')
    parser.add_argument("-o", "--opt", default='normal', help="optimizer")
    parser.add_argument("-a", "--archive", default=15, help="archive index")
    args = parser.parse_args()

    task = args.task
    n = args.num
    budget = 0

    root_dir = os.path.join("./result", f"archive-{args.archive}")

    if not os.path.exists(os.path.join(root_dir, task, f"{n}")):
        os.mkdir(os.path.join(root_dir, task, f"{n}"))

    upb_suffix = '2'

    opt = args.opt

    assert opt in ['normal', 'modified1', 'modified2', 'multilinear', 'multilinear2', 'matroid']

    Y_p = "max"

    constraint_count = 4

    for seed in range(90, 100):
        for budget in range(16, 17):
            start = time.time()

            if opt == 'matroid':
                model = model_factory(task, n, seed, budget, cm="normal", knap=True, enable_packing=False,
                                      constraint_count=constraint_count)
                model.enable_matroid()
                S, upb, w = greedy_for_matroid(model=model, opt_type=opt)
            else:
                model = model_factory(task, n, seed, budget, cm="normal", knap=True, enable_packing=True,
                                      constraint_count=constraint_count)
                model.bv = np.array([budget] * constraint_count)
                S, upb, w = MWU(model, upb='ub0', upb_function_mode='none', opt_type = opt)

            stop = time.time()

            af = float(model.objective(list(S)) / upb)
            # assert af <= 1.0

            final_res = {
                "S": S,
                "f(S)": model.objective(list(S)),
                "ub0": upb,
                "AF": float(model.objective(list(S)) / upb),
                "worst AF": w,
                "time_opt": stop-start,
            }

            print(f"Done:seed:{seed}, budget:{budget}, result:{final_res}")

            save_dir = os.path.join(root_dir, task, f"{n}", f"{seed}")

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_path = os.path.join(save_dir, "{}-{}-{}-{:.2f}-{}.pckl".format(
                f"mwu", opt, model.__class__.__name__, budget, Y_p))

            with open(save_path, "wb") as wrt:
                pickle.dump(final_res, wrt)
