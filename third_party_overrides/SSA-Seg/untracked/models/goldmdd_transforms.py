import random

import cv2
import numpy as np
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GoldMDDGaussianBlur:
    """Apply Gaussian blur to RGB image only."""

    def __init__(self, prob=0.2, sigma_min=0.3, sigma_max=1.2):
        self.prob = float(prob)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __call__(self, results):
        if random.random() >= self.prob:
            return results

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        ksize = max(3, int(2 * round(3.0 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        results["img"] = cv2.GaussianBlur(results["img"], (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(prob={self.prob}, "
            f"sigma_min={self.sigma_min}, sigma_max={self.sigma_max})"
        )


@PIPELINES.register_module()
class GoldMDDRandomRotate90:
    """Rotate image + label by k * 90 degrees with k in [0,1,2,3]."""

    def __call__(self, results):
        k = random.randint(0, 3)
        if k == 0:
            return results

        results["img"] = np.ascontiguousarray(np.rot90(results["img"], k, axes=(0, 1)))
        for key in results.get("seg_fields", []):
            results[key] = np.ascontiguousarray(np.rot90(results[key], k, axes=(0, 1)))
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"
