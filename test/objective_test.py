import random
import time
import unittest

import numpy as np

from compute_exp import upper_bounds
from image_sum import ImageSummarization
from movie_recommendation import MovieRecommendation
from revenue_max import RevenueMax


class ObjectiveTest(unittest.TestCase):
    def test_img_sum(self):
        ground_set_num = max_num = 50
        model = ImageSummarization(
            image_path="../dataset/image/500_cifar10_sample.npy", budget=100, max_num=max_num)

        print(model.objective([1, 34, 17, 18, 19]))
        print(model.objective([1, 34, 17, 18]))
        print(model.objective([1, 34, 18]))
        print(model.objective([1, 34]))

        for i in range(0, 100):
            random.seed(time.time())
            a = random.sample(range(0, ground_set_num), 20)
            b = random.sample(range(0, ground_set_num), 20)
            union_set = list(set(a) | set(b))
            intersection_set = list(set(a) & set(b))
            left = model.objective(a) + model.objective(b)
            right = model.objective(union_set) + model.objective(intersection_set)

            print(f"Left:{left}, Right:{right}, Diff:{left-right}")
            self.assertGreaterEqual(left, right)


    def test_movie_recommendation(self):
        ground_set_num = max_num = 50
        model = MovieRecommendation(
            matrix_path="../dataset/movie/user_by_movies_small_rating.npy", budget=100, k=30, n=50)
        for i in range(0, 100):
            random.seed(time.time())
            a = random.sample(range(0, ground_set_num), 20)
            b = random.sample(range(0, ground_set_num), 20)
            union_set = list(set(a) | set(b))
            intersection_set = list(set(a) & set(b))
            left = model.objective(a) + model.objective(b)
            right = model.objective(union_set) + model.objective(intersection_set)

            print(f"Left:{left}, Right:{right}, Diff:{left-right}")
            self.assertGreaterEqual(left, right)

    def test_revenue_max(self):
        ground_set_num = max_num = 50
        model = RevenueMax(budget=1.0, pckl_path="../dataset/revenue/25_youtube_top5000.pkl")
        for i in range(0, 100):
            random.seed(time.time())
            a = random.sample(range(0, ground_set_num), 20)
            b = random.sample(range(0, ground_set_num), 20)
            union_set = list(set(a) | set(b))
            intersection_set = list(set(a) & set(b))
            left = model.objective(a) + model.objective(b)
            right = model.objective(union_set) + model.objective(intersection_set)

            print(f"Left:{left}, Right:{right}, Diff:{left-right}")
            self.assertGreaterEqual(left, right)


if __name__ == '__main__':
    unittest.main()
