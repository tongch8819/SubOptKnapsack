import heapq


class HeapObj(object):
    def __init__(self, s, v):
        self.s = s
        self.v = v

    def __lt__(self, other):
        return self.v > other.v

    def __eq__(self, other):
        return self.v == other.v

    def __str__(self):
        return f"{self.s}, {self.v}"


class MaxHeap(object):
    def __init__(self):
        self.h = []

    def push(self, v):
        heapq.heappush(self.h, v)

    def pop(self):
        return heapq.heappop(self.h)

    def __getitem__(self, item):
        assert item < len(self.h)
        return self.h[item]

    def clear(self):
        self.h = []

    def size(self):
        return len(self.h)
