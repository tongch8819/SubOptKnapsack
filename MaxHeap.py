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
