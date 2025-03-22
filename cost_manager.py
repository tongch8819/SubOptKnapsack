
class CostManager:
    def __init__(self):
        self.mode = None
        self.model = None
        self.assign = None

        self.minimal_cost = 1.6
        self.maximal_cost = 5.6

        pass

    def set_mode(self, mode):
        self.mode = mode

    def set_model(self, model):
        self.model = model

    def build(self):
        assert self.mode is not None

        if self.mode == 'normal':
            self.assign = self.assign_random
        elif self.mode == 'positive':
            self.assign = self.assign_positive
        elif self.mode == 'negative':
            self.assign = self.assign_negative
        else:
            raise Exception(f"Mode {self.mode} does not exist.")

        pass

    def assign_random(self):
        pass

    def assign_positive(self):
        min_idx, min_v = None, 0
        max_idx, max_v = None, 0

        for i in range(0, len(self.model.ground_set)):
            if min_idx is None or min_v > self.model.objective([i]):
                min_idx, min_v = i, self.model.objective([i])

            if max_idx is None or max_v < self.model.objective([i]):
                max_idx, max_v = i, self.model.objective([i])

        factor = (max_v - min_v) / (self.maximal_cost - self.minimal_cost)

        costs = []
        for i in range(0, len(self.model.ground_set)):
            costs.append(self.minimal_cost + (self.model.objective([i]) - min_v) / factor)

        # print(f"c:{costs[:10]}, {[self.model.objective([i]) for i in range(0, 10)]}")

        return costs

    def assign_negative(self):
        min_idx, min_v = None, 0
        max_idx, max_v = None, 0

        for i in range(0, len(self.model.ground_set)):
            if min_idx is None or min_v > self.model.objective([i]):
                min_idx, min_v = i, self.model.objective([i])

            if max_idx is None or max_v < self.model.objective([i]):
                max_idx, max_v = i, self.model.objective([i])

        factor = (max_v - min_v) / (self.maximal_cost - self.minimal_cost)

        costs = []
        for i in range(0, len(self.model.ground_set)):
            costs.append(self.maximal_cost - (self.model.objective([i]) - min_v) / factor)

        # print(f"c:{costs[:10]}, {[self.model.objective([i]) for i in range(0, 10)]}")

        return costs

