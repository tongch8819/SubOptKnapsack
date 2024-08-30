import random
from typing import List, Set


class Matroid:
    def __init__(self, ground_set, rank, basis_count):
        self.bases = []
        self.ground_set = ground_set
        self.rank = rank
        self.basis_count = basis_count

        self.__random_generate()
        pass

    def __random_generate(self):
        count = 0

        while count < self.basis_count:
            temp = set(random.sample(self.ground_set, self.rank))
            if temp not in self.bases:
                self.bases.append(temp)
                count += 1

    def is_legal(self, s: Set[int]) -> bool:
        for basis in self.bases:
            if s.issubset(basis):
                return True
        return False




