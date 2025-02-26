from base_task_usm_mc import BaseTask_USMMC
import numpy as np
import os
from typing import List


class MovieRecommendation_USMMC(BaseTask_USMMC):
    def __init__(self, k: int = None, n: int = None, sim_type: str = "cosine", matrix_path: str = None, llambda: float = 0.5):
        """
        Inputs:
        - k: number of users
        - n: number of movies
        - b: budget

        The objective is non-negative and non-monotone.
        """
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
        self.similarity_type = sim_type  # or cosine

        avg_ratings = np.average(self.M, axis=0)
        assert min(avg_ratings) >= 0 and max(
            avg_ratings) <= 10, "Average rating should lie in [0, 10]"
        self.costs_obj = 10 - avg_ratings

        # factor in objective
        self.llambda = llambda
        assert 0. <= self.llambda <= 1.

    @property
    def ground_set(self):
        return self.movies
    
    @ground_set.setter
    def ground_set(self, v):
        self.movies = v

    def load_matrix(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.npy does not exist.")
        return np.load(path)

    def similarity(self, u, v):
        """
        Inputs:
        - u: some movie
        - v: some movie
        """
        u_vec, v_vec = self.M[:, u], self.M[:, v]
        if self.similarity_type == 'inner':
            return np.dot(u_vec, v_vec)
        elif self.similarity_type == 'cosine':
            return np.dot(u_vec, v_vec) / (np.linalg.norm(u_vec) * np.linalg.norm(v_vec))
        elif self.similarity_type == 'exp':
            euclidean_dist = np.linalg.norm(u_vec - v_vec)
            llambda = 2.
            return np.exp(- llambda * euclidean_dist)
        else:
            raise ValueError("Unsupported similarity type.")

    def first_part(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        first = 0.
        second = 0.
        S = set(S)
        for v in S:
            for u in self.ground_set:
                s_uv = self.similarity(u, v)
                first += s_uv
                if u in S:  # search time: O(1)
                    second += s_uv

        scaler = 1
        return (first - self.llambda * second) / scaler
    
def construct_model():
    return MovieRecommendation_USMMC(matrix_path="../dataset/movie/user_by_movies_small_rating.npy", k=40, n=40)

def main():
    model = MovieRecommendation_USMMC(matrix_path="../dataset/movie/user_by_movies_small_rating.npy")

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("g(S) =", model.objective(S))


if __name__ == "__main__":
    main()
