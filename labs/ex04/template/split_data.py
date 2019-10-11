# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # source: https://stackoverflow.com/a/3677283
    indices = np.random.permutation(x.shape[0])
    rate = int(np.floor(indices.shape[0] * ratio))
    training_idx, test_idx = indices[:rate], indices[rate:]
    training = (x[training_idx], y[training_idx])
    test = (x[test_idx], y[test_idx])
    return training, test
