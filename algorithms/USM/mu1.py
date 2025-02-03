from base_task_usm_mc import BaseTask_USMMC
import movie_recommendation

from copy import deepcopy
import logging
logging.basicConfig(level=logging.DEBUG)
import random

class DataDependentUpperboundDoubleGreedyIterativePrune:
    def __init__(self, model: BaseTask_USMMC):
        self.model = model
        self.objective = model.objective
        self.ground_set = model.ground_set

    def compute(self):
        # Sl: output of deterministic double greedy algorithm
        Sl = self.deterministicDoubleGreedy()
        fv_Sl = self.objective(Sl)

        # A_B_pairs: output of iterative prune
        A_B_pairs = self.iterativePrune()
        fv_An = self.objective(A_B_pairs[0])
        fv_Bn = self.objective(A_B_pairs[1])

        mu1 = 3 * fv_Sl - (fv_An + fv_Bn)
        logging.debug(f"fv_Sl: {fv_Sl}, fv_An: {fv_An}, fv_Bn: {fv_Bn}, mu1: {mu1}")
        return mu1

    def iterativePrune(self):
        maximum_iterations = 1e6
        prev_A, A = [], []
        prev_B, B = deepcopy(self.ground_set), deepcopy(self.ground_set)
        def is_set_equivalent(A, B):
            return set(A) == set(B)
        # brute force implementation
        i = 0
        while not is_set_equivalent(prev_A, A) or not is_set_equivalent(prev_B, B):
            prev_A = deepcopy(A)
            prev_B = deepcopy(B)
            A = filter(lambda x: self.model.marginal_gain(x, set(prev_B) - set([x])) >= 0, A)
            B = filter(lambda x: self.model.marginal_gain(x, prev_A) >= 0, B)
            i += 1
            if i > maximum_iterations:
                logging.warning("Maximum iterations reached")
                break
        return A, B

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
    
    def randomizedDoubleGreedy(self):
        return self.baseDoubleGreedy([], self.ground_set, randomizded=True)

def main():
    # unit test for mu1 computation
    model = movie_recommendation.construct_model()

    upb_double_greedy = DataDependentUpperboundDoubleGreedyIterativePrune(model).compute()
    print("mu1 =", upb_double_greedy)

if __name__ == '__main__':
    main()