from movie_recommendation import MovieRecommendation
from mgreedy_nonmono import modified_greedy
from greedymax_nonmono import greedy_max
from algorithms.set_enumeration import set_enumeration_parallel

import numpy as np
import pandas as pd
from functools import partial


three_set_enumeration = partial(set_enumeration_parallel, alg_handler = modified_greedy, num_initial_elements = 3)
two_set_enumeration = partial(set_enumeration_parallel, alg_handler = modified_greedy, num_initial_elements = 2)

data_path = "/home/ctong/Projects/SubOptKnapsack/" + \
    "dataset/movie/user_by_movies_small_rating.npy"
upperbound_method_name = "ub1"  # singleton value back fill
num_of_movies = 50
num_of_users = 50
name_algo_pair_lst = [
    # ("3setenum", three_set_enumeration),
    ("2setenum", two_set_enumeration),
    ("mgreedy", modified_greedy),
    ("greedymax", greedy_max),
]
exp_num = "exp7"
res_dump_path = "/home/ctong/Projects/SubOptKnapsack/result/nonmono/{}".format(exp_num)
fixed_budget = 40.0
fixed_lambda = 0.75
num_of_samples = 5


def lambda_exp(verbose=False):
    """
    Varying lambda to see the difference given fixed budget
    """
    lambdas_arr = np.linspace(0.5, 1, num_of_samples)
    procs = []

    for name, algo_handle in name_algo_pair_lst:
        data = []
        for llambda in lambdas_arr:
            # average cost of each movie is around 5
            # set it to be non-monotone submodular function
            model = MovieRecommendation(matrix_path=data_path,
                                        k=num_of_users, n=num_of_movies, budget=fixed_budget, llambda=llambda, is_mono=False)
            
            # cur_proc = Process(target=algo_handle, args=(model, ), kwargs={'upb': upperbound_method_name})
            # procs.append(cur_proc)
            # cur_proc.start()

            res = algo_handle(model, upb=upperbound_method_name)
            res['llambda'] = llambda
            # res_table.append(res)
            # df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
            # df = df.append(res, ignore_index=True)
            data.append(res)
        df = pd.DataFrame(data, columns=['llambda', 'S', 'f(S)', 'c(S)', 'Upb', 'AF'])

        dump_path = res_dump_path + f'/nonmono_exp_lambda_{name}.csv'
        df.to_csv(dump_path)
        if verbose:
            print("Dump file: {}".format(dump_path))


def budget_exp(verbose=False):
    """
    Varying budget to see the difference given fixed lambda
    """
    budget_arr = np.linspace(30, 50, num_of_samples)

    for name, algo_handle in name_algo_pair_lst:
        data = []
        for budget in budget_arr:
            # average cost of each movie is around 5
            # set it to be non-monotone submodular function
            model = MovieRecommendation(matrix_path=data_path,
                                        k=num_of_users, n=num_of_movies, budget=budget, llambda=fixed_lambda, is_mono=False)
            res = algo_handle(model, upb=upperbound_method_name)
            res['budget'] = budget
            # res_table.append(res)
            # df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
            # df = df.append(res, ignore_index=True)
            data.append(res)

        df = pd.DataFrame(data, columns=['budget', 'S', 'f(S)', 'c(S)', 'Upb', 'AF'])
        dump_path = res_dump_path + f'/nonmono_exp_budget_{name}.csv'
        df.to_csv(dump_path)
        if verbose:
            print("Dump file: {}".format(dump_path))



def main():
    verbose = True
    # lambda_exp(verbose)
    budget_exp(verbose)

if __name__ == "__main__":
    main()

