import os
import pickle
import time

from MWU import MWU
from compute_knapsack_exp import model_factory
import numpy as np

if __name__ == "__main__":
    task = "facebook"
    n = 500
    budget = 0

    root_dir = os.path.join("./result", "archive-14")
    if not os.path.exists(os.path.join(root_dir, task, f"{n}")):
        os.mkdir(os.path.join(root_dir, task, f"{n}"))

    upb_suffix = '0'

    opt = ''
    if upb_suffix == '0':
        opt = 'normal'
    elif upb_suffix == '2':
        opt = 'modified'
    upb_function_mode = 'none'
    Y_p = "max"

    constraint_count = 4

    for seed in range(125, 200):
        for budget in range(6, 20):
            start = time.time()

            model = model_factory(task, n, seed, budget, cm="normal", knap=True, enable_packing=True, constraint_count = constraint_count)
            model.bv = np.array([budget] * constraint_count)

            S, upb, w = MWU(model, upb='ub0', upb_function_mode=upb_function_mode, opt_type = opt)

            stop = time.time()

            af = float(model.objective(list(S)) / upb)
            assert af <= 1.0

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
                f"mwu{upb_suffix}", upb_function_mode, model.__class__.__name__, budget, Y_p))

            with open(save_path, "wb") as wrt:
                pickle.dump(final_res, wrt)
