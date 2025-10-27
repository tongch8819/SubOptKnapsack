from OptimalAlg import OptimalAlg
from base_task import BaseTask


class IDAstar(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

    def ida(self):
        pass

    def g(self):
        pass

    def h(self):
        pass

    def is_goal(self, node):
        pass

    def search(self):
        pass

    def build(self):
        pass

    def optimize(self):
        pass
