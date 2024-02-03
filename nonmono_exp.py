from movie_recommendation import MovieRecommendation
from mgreedy_nonmono import modified_greedy_ub1
from greedymax_nonmono import greedy_max_ub1
import numpy as np
import pandas as pd

data_path =  "/home/ctong/Projects/SubOptKnapsack/" + "dataset/movie/user_by_movies_small_rating.npy"
lambdas_arr = np.linspace(0.5, 1, 5)

df = pd.DataFrame(columns=['llambda', 'S', 'f(S)', 'c(S)', 'Upb', 'AF'])
for llambda in lambdas_arr:
    # average cost of each movie is around 5
    # set it to be non-monotone submodular function
    model = MovieRecommendation(matrix_path=data_path, 
                                k=50, n=50, budget=100.0, llambda=llambda, is_mono=False)
    res = modified_greedy_ub1(model)
    res['llambda'] = llambda
    # res_table.append(res)
    df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)

df.to_csv('nonmono_exp_mgreedy.csv')


df = pd.DataFrame(columns=['llambda', 'S', 'f(S)', 'c(S)', 'Upb', 'AF'])
for llambda in lambdas_arr:
    # average cost of each movie is around 5
    # set it to be non-monotone submodular function
    model = MovieRecommendation(matrix_path=data_path, 
                                k=50, n=50, budget=100.0, llambda=llambda, is_mono=False)
    res = greedy_max_ub1(model)
    res['llambda'] = llambda
    # res_table.append(res)
    df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)

df.to_csv('nonmono_exp_greedymax.csv')