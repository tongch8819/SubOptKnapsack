import random
import time
import unittest

import numpy as np

from dblp_graph_coverage import DblpGraphCoverage
from facebook_graph_coverage import FacebookGraphCoverage
from image_sum import ImageSummarization
from movie_recommendation import MovieRecommendation
from revenue_max import RevenueMax, CalTechMaximization
import networkx as nx
from feature_selection import AdultIncomeFeatureSelection, SensorPlacement
from influence_maximization import YoutubeCoverage, CitationCoverage


class GraphCoverageTest(unittest.TestCase):
    def test_load(self):

        # adult = AdultIncomeFeatureSelection(1,100,"../dataset/adult-income", construct_graph=True, cost_mode="big")
        # # # print(len(adult.samples))
        # #
        # sensor = SensorPlacement(1,100,"../dataset/berkley-sensor", construct_graph=True, cost_mode="big")
        #
        youtube = YoutubeCoverage(0, 1045, "../dataset/com-youtube", knapsack=True, construct_graph=True)
        #
        # cal = CalTechMaximization(0, 25, "../dataset/caltech", knapsack=True, prepare_max_pair=False, construct_graph=True, graph_suffix="-25")
        # citation = CitationCoverage(0, 1000, "../dataset/cite-HepPh", knapsack=True, prepare_max_pair=False, construct_graph=True, cost_mode="big")
        #
        # nodes = random.sample(list(intact_graph.nodes), self.max_nodes)

        # facebook = FacebookGraphCoverage(0, 100, graph_path="../dataset/facebook", knapsack=True,prepare_max_pair=False,print_curvature=False, construct_graph = True, graph_suffix="-100")
        #
        # dblp = DblpGraphCoverage(0, n=5000, graph_path="../dataset/com-dblp", knapsack=True,prepare_max_pair=False,print_curvature=False, construct_graph = True)


        # citation = CitationCoverage(0, 1000, "../dataset/cite-HepPh", knapsack=True, prepare_max_pair=False, construct_graph=True)



#graph_task = FacebookGraphCoverage(1, n=5000, graph_path="../dataset/facebook/facebook_combined.txt")
        #print(graph_task.costs_obj)
        #print(graph_task.objective([51]))

if __name__ == '__main__':
    unittest.main()
