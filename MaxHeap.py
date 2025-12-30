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
    def __init__(self, s, v=None, cost=None, candidate=None, w=0, first_child=False, heuristic_sequence=None, max_idx=0,
                 visited=False):
        super().__init__(s, v, cost, candidate=candidate, w=w, max_idx=max_idx, visited=visited)

        self.first_child = first_child
        self.heuristic_sequence = heuristic_sequence


class BranchAndBoundNode(HeapObj):
    # s: current set
    # c: candidate set
    # w: remaining budget
    def __init__(self, s, c, w):
        super().__init__(s=s, candidate=c, w=w)


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


def find_parent(idx):
    return int(idx / 2)


class SimpleMaxHeap(object):
    def __init__(self):
        self.elements = []
        self.h = []

        self.current_size = 0

    def push(self, node):
        ele_idx = len(self.elements)
        self.elements.append(node)

        self.h.append(ele_idx)

        idx_idx = self.current_size
        while idx_idx > 0:
            parent_idx_idx = find_parent(idx_idx)
            if self.elements[self.h[parent_idx_idx]].v < self.elements[self.h[idx_idx]].v:
                t = self.h[parent_idx_idx]
                self.h[parent_idx_idx] = self.h[idx_idx]
                self.h[idx_idx] = t
                idx_idx = parent_idx_idx
            else:
                break

        self.current_size = self.current_size + 1

    def pop(self):
        assert len(self.h) > 0, "Heap is empty."
        idx_idx = self.current_size - 1
        self.h[0] = self.h[idx_idx]

        idx_idx = 0
        while 2 * idx_idx <= self.current_size:
            large_child = idx_idx * 2
            small_child = idx_idx * 2 + 1

            if large_child < self.current_size and self.elements[self.h[large_child]].v < self.elements[
                self.h[small_child]].v:
                large_child += 1

            if self.elements[self.h[large_child]].v <= self.elements[self.h[idx_idx]].v:
                break

            t = self.h[large_child]
            self.h[large_child] = self.h[idx_idx]
            self.h[idx_idx] = t

            idx_idx = large_child

        self.current_size = self.current_size - 1

    def remove(self, ele):
        pass

    def top(self):
        assert len(self.h) > 0, "Heap is empty."

        return self.elements[self.h[0]]

    def __getitem__(self, item):
        assert item < len(self.h)
        return self.h[item]

    def clear(self):
        pass

    def size(self):
        pass
