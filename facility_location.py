import random

from base_task import BaseTask
import numpy as np
import os
from typing import Set, List


class FacilityLocation(BaseTask):
    """
    Movie Recommendation with Facility Location Objective
    under CC (Cardinality Constraint)

    Implements using SubOpt knapsack framework.

    # TODO: the objective is a modular function
    """

    def __init__(self, matrix_path, k: int, construct_graph = False):
        self.M = self.load_matrix(matrix_path)  # movie by user matrix

        self.num_users = self.M.shape[1]
        self.num_movies = self.M.shape[1]
        self.b = k
        self.movies = [i for i in range(self.num_movies)]
        self.costs_obj = [1] * self.num_movies

    @property
    def ground_set(self):
        return self.movies

    def load_matrix(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.npy does not exist.")
        return np.load(path)

    def internal_objective(self, S: List[int]):
        if type(S) is not list:
            S = list(S)
        if len(S) == 0:
            return 0.
        # compute maximum across all users
        return self.M[S, :].max(axis=1).sum()


class MovieFacilityLocation(BaseTask):
    def __init__(self, budget: float, k: int = None, n: int = None, seed = 21, sim_type: str = "cosine", matrix_path: str = None,
                 llambda: float = 0.5, knapsack=True, prepare_max_pair=True, print_curvature=False):
        """
        Inputs:
        - k: number of users
        - n: number of movies
        - b: budget

        The objective is non-negative and non-monotone.
        """

        super().__init__()
        if matrix_path is None:
            # self.M[i][j] denotes the rating of user i for movie j
            self.num_users = k
            self.num_movies = n
            self.M = np.random.random(size=(k, n))
        else:
            self.M = self.load_matrix(matrix_path)
            self.M = self.M[:k, :n]   # use k and n to truncate
            self.num_users, self.num_movies = self.M.shape
        self.movies = [i for i in range(self.num_movies)]
        self.users = [i for i in range(self.num_users)]
        self.similarity_type = sim_type  # or cosine

        avg_ratings = np.average(self.M, axis=0)
        assert min(avg_ratings) >= 0 and max(
            avg_ratings) <= 10, "Average rating should lie in [0, 10]"
        self.costs_obj = 10 - avg_ratings

        # factor in objective
        self.llambda = llambda
        assert 0. <= self.llambda <= 1.

        self.budget = budget
        self.knapsack = knapsack

        if prepare_max_pair:
            self.prepare_max_2_pair()

        if print_curvature:
            self.print_curvature()

    @property
    def ground_set(self):
        return self.movies

    def load_matrix(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.npy does not exist.")
        return np.load(path)


    def internal_objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        S = list(S)
        if len(S) == 0:
            return 0
        sum = 0.
        for user in self.users:
            sum += max(self.M[:, S][user])
        return sum

    def cost_of_set(self, S: List[int]):
        if not self.knapsack:
            return len(S)
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        if not self.knapsack:
            return 1
        return self.costs_obj[singleton]

def main():
    model = FacilityLocation(
        matrix_path="dataset/movie/movie_by_user_small_rating_rank_norm.npy", k=4)

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
