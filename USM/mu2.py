from base_task_usm_mc import BaseTask_USMMC
import movie_recommendation

from copy import deepcopy
import logging
logging.basicConfig(level=logging.DEBUG)
import random
import math

class DataDependentUpperboundROIGreedy:
    def __init__(self, model: BaseTask_USMMC):
        self.model = model
        self.objective = model.objective
        self.ground_set = model.ground_set

    def compute(self):
        # Sl: output of deterministic double greedy algorithm
        Sl = self.ROIGreedy()
        fv_Sl = self.objective(Sl)

        # A_B_pairs: output of iterative prune
        cV = self.model.cost_of_set(self.ground_set)

        mu_1 = self.data_dependent_upb_USM()
        try:
            mu = fv_Sl + cV * math.log(mu_1 + 1e-5) + math.exp(-1)
            logging.debug(f"fv_Sl: {fv_Sl}, cV: {cV}, mu_1: {mu_1}")

            # debugging 
            cSl = self.model.cost_of_set(Sl)
            potential_mu = fv_Sl + cSl * math.log(mu_1 + 1e-5) + math.exp(-1)
            logging.debug(f"c(S_l): {cSl}, potential mu: {potential_mu}")
        except Exception as e:
            logging.error(f"An error occurred during computation: {e}")
            logging.error(f"fv_Sl: {fv_Sl}, cV: {cV}, mu_1: {mu_1}")
        return mu

    def ROIGreedy(self):
        S = []
        while True:
            optimal_element, max_marginal_gain = None, None
            for v in self.ground_set:
                if v in S:
                    continue
                if max_marginal_gain is None or self.model.marginal_gain(v, S) >= max_marginal_gain:
                    optimal_element = v
                    max_marginal_gain = self.model.marginal_gain(v, S)
            if max_marginal_gain <= 0:
                break
            S.append(optimal_element)
        return S


    def baseDoubleGreedy(self, S0, T0, randomizded=False):
        S = set(deepcopy(S0))
        T = set(deepcopy(T0))
        for v in self.ground_set:
            lhs = self.model.marginal_gain(v, S)
            rhs = - self.model.marginal_gain(v, T - set([v]))
            if not randomizded:
                can_append_S = (lhs >= rhs)
            else:
                ratio = (lhs / (lhs + rhs))
                can_append_S = random.random() <= ratio
            
            if can_append_S:
                S.add(v)
            else:
                T.remove(v)
        assert len(S) == len(T)
        return S

    def deterministicDoubleGreedy(self):
        return self.baseDoubleGreedy([], self.ground_set, randomizded=False)
    
    def data_dependent_upb_USM(self):
        Sl = self.deterministicDoubleGreedy()
        fv_Sl = self.objective(Sl)
        return 2 * fv_Sl

def main():
    # unit test for mu1 computation
    model = movie_recommendation.construct_model()
    upb_roi = DataDependentUpperboundROIGreedy(model).compute()
    print("Upperbound ROI Greedy:", upb_roi)
    

if __name__ == '__main__':
    main()