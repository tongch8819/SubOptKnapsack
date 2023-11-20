import random
import time
import unittest

import numpy as np

from compute_exp import upper_bounds
from dblp_graph_coverage import DblpGraphCoverage
from facebook_graph_coverage import FacebookGraphCoverage
from image_sum import ImageSummarization
from movie_recommendation import MovieRecommendation
from revenue_max import RevenueMax


class GraphCoverageTest(unittest.TestCase):
    def test_load(self):
        graph_task = DblpGraphCoverage(1, n=5000, graph_path="../dataset/com-dblp/com-dblp.top5000.cmty.txt")
        print(graph_task.graph)
        graph_task = FacebookGraphCoverage(1, n=5000, graph_path="../dataset/facebook/facebook_combined.txt")
        print(graph_task.costs_obj)

if __name__ == '__main__':
    unittest.main()
