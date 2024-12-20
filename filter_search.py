from OptimalAlg import OptimalAlg
from base_task import BaseTask
from MaxHeap import MaxHeap, HeapObj


class FS(OptimalAlg):
    def __init__(self, model: BaseTask):
        super().__init__(model)

        self.closed_list = set()
        self.heap = MaxHeap()

        self.E = set()

    def build(self):
        self.closed_list.clear()
        self.heap.clear()

    def f(self, S):
        pass

    # the heuristic function
    def h(self, S):
        pass

    def g(self, S):
        return self.f(S) + self.alpha * self.h(S)

    def optimize(self):
        self.heap.push(HeapObj({}, self.g({})))
        while self.heap.size() > 0:
            obj = self.heap.pop()
            s, v = obj.s, obj.v
            if self.h(s) == 0:
                return s
            if s not in self.closed_list:
                self.closed_list.add(s)
            for ele in self.model.ground_set - s:
                s_plus = s | {ele}
                self.heap.push(HeapObj(s_plus, self.g(s_plus)))
