import unittest

from MaxHeap import SimpleMaxHeap


class Node:
    def __init__(self, v):
        self.v = v


class TestHeap(unittest.TestCase):
    def test_1(self):
        heap = SimpleMaxHeap()
        heap.push(Node(10))
        heap.push(Node(5))
        heap.pop()

        self.assertEqual(heap.top().v, 5)
        self.assertEqual(heap.size(), 1)


if __name__ == '__main__':
    unittest.main()
