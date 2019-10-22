# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_error(y, tx, w):
    return y - np.dot(tx, w)

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)

def least_squares(y, tx):
    """Least squares normal equations."""
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = calculate_mse(compute_error(y, tx, w))
    return w
