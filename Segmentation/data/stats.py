import numpy as np

from data.dataset import Brats184


dataset = Brats184(None, None, None)

label_counts = {i: 0 for i in range(4)}
for _, target in dataset:
    vals, counts = np.unique(target, return_counts=True)
    for v, c in zip(vals, counts):
        label_counts[v] += c
