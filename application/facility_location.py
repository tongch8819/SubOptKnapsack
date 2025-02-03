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

    def __init__(self, matrix_path, k: int):
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

    def objective(self, S: List[int]):
        if type(S) is not list:
            S = list(S)
        if len(S) == 0:
            return 0.
        # compute maximum across all users
        return self.M[S, :].max(axis=1).sum()


def main():
    model = FacilityLocation(
        matrix_path="dataset/movie/movie_by_user_small_rating_rank_norm.npy", k=4)

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
