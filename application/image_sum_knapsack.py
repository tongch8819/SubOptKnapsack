from application.base_task import BaseTask
from application.constraint import KnapsackConstraint

import os
import numpy as np
from typing import List, Set


class ImageSummarizationKnapsack(BaseTask):
    """
    The goal is to identify a representative subset S from a large collection V of images under a limited budget

    Make sure image is normalized before loading.

    The objective is non-negative and non-monotone.
    """

    def __init__(self, budget_ratio: float, image_path: str, max_num : int = None):
        super().__init__(is_mono=False)
        self.images = self.load_images(image_path)
        if max_num is not None:
            self.images = self.images[:max_num, :]
        n = self.images.shape[0]
        self._objects = [i for i in range(n)]

        assert self.images.shape == (n, 3072)

        # the costs are normalized s.t. the average cost is 1/10
        self.cost_obj = [1 / self._rms_contrast(i) for i in range(n)]
        self.constraint = KnapsackConstraint(cost_obj=self.cost_obj, budget_ratio=budget_ratio)


    @property
    def ground_set(self):
        return self._objects
    
    @property
    def budget(self):
        return self.b

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
        if len(S) == 0:
            return 0.
        t1 = t2 = 0.
        for u in self.ground_set:
            max_s = max([
                self.similarity(u, v) for v in S
            ])
            t1 += max_s
        for u in S:
            for v in S:
                t2 += self.similarity(u, v)
        t3 = 1 / len(self.ground_set) * t2
        # print(f"S: {S}, t1: {t1}, t3: {t3}")
        return t1 - t3

    def similarity(self, u: int, v: int):
        """
        The similarity s(u,v) is computed as the cosine similarity of the 
        3072-dimensional pixel vectors of image u and image v.
        """
        u_img_vec, v_img_vec = self.images[u], self.images[v]
        # u_img_vec = np.flatten(u_img)
        # v_img_vec = np.flatten(v_img)
        cos_sim = np.dot(u_img_vec, v_img_vec) / \
            (np.linalg.norm(u_img_vec) * np.linalg.norm(v_img_vec))
        return cos_sim

        # maximum absolute difference
        # return np.max(np.abs(u_img_vec - v_img_vec))

    

def main():
    model = ImageSummarizationKnapsack(budget_ratio=0.3,
        image_path="application/dataset/image/500_cifar10_sample.npy", max_num=10)
    # for i in model.ground_set:
    #     fs = model.cutout_marginal_gain(i, set(model.ground_set))
    #     print(f"f( {i} | V) = {fs}")
    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.constraint.cost_of_set(S))
    print("budget b = ", model.constraint.budget)


if __name__ == "__main__":
    main()
