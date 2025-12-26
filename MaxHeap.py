import heapq


class HeapObj(object):
    def __init__(self, s, v=None, cost=None, candidate=None, w=None, max_idx=0, visited=False):
        self.s = s
        self.cost = cost
        self.candidate = candidate
        self.budget = w
        self.max_idx = max_idx
        self.visited = visited

        self.v = v

    def __lt__(self, other):
        return self.v > other.v

    def __eq__(self, other):
        return self.v == other.v

    def __str__(self):
        return f"{self.s}, {self.v}"


class EfficientBFSHeapObj(HeapObj):
    def __init__(self, s, v=None, cost=None, candidate=None, w=0, first_child=False, heuristic_sequence=None, max_idx=0, visited=False):
        super().__init__(s, v, cost, candidate=candidate, w=w, max_idx=max_idx, visited=visited)

        self.first_child = first_child
        self.heuristic_sequence = heuristic_sequence


class MaxHeap(object):
    def __init__(self):
        self.h = []

    def push(self, v):
        heapq.heappush(self.h, v)

    def pop(self):
        return heapq.heappop(self.h)

    def remove(self, ele):
        try:
            self.h.remove(ele)
            heapq.heapify(self.h)
        except ValueError:
            print(f"Element {ele} not found in heap.")

    def top(self):
        return self.h[0]

    def __getitem__(self, item):
        assert item < len(self.h)
        return self.h[item]

    def clear(self):
        self.h = []

    def size(self):
        return len(self.h)
