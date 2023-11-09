import os
import numpy as np
from typing import List, Set


class ImageSummarization:
    """
    The goal is to identify a representative subset S from a large collection V of images under a limited budget

    Make sure image is normalized before loading.
    """

    def __init__(self, n: int, image_path: str):
        self._objects = [i for i in range(n)]
        self.images = self.load_images(image_path)
        assert self.images.shape == (n, 3072)

        self.costs_obj = [1 / self._rms_contrast(i) for i in range(n)]

    @property
    def ground_set(self):
        return self._objects

    def _rms_contrast(self, u: int):
        # The RMS contrast is defined as the standard deviation of the normalized pixel intensity values.
        u_img = self.images[u]
        return u_img.std()

    def load_images(self, path):
        if not os.path.isfile(path):
            raise OSError("File *.npy does not exist.")
        return np.load(path)

    def objective(self, S: List[int]):
        # t1: coverage
        # t2: penalty factor
        t1 = t2 = 0.
        for u in self.ground_set:
            max_s = max([
                self.similarity(u, v) for v in S
            ])
            t1 += max_s
        for u in S:
            for v in S:
                t2 += self.similarity(u, v)

        return t1 - 1 / len(self.ground_set) * t2

    def similarity(self, u: int, v: int):
        u_img_vec, v_img_vec = self.images[u], self.images[v]
        # u_img_vec = np.flatten(u_img)
        # v_img_vec = np.flatten(v_img)
        cos_sim = np.dot(u_img_vec, v_img_vec) / \
            (np.linalg.norm(u_img_vec) * np.linalg.norm(v_img_vec))
        return cos_sim

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]


def main():

    model = ImageSummarization(
        n=500, image_path="dataset/image/500_cifar10_sample.npy")

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
