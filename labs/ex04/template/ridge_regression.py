# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""
from costs import compute_mse
import numpy as np

def ridge_regression(y, tx, lambda_):
    """Ridge regression equations."""
    N = y.shape[0]
    D = tx.shape[1]
    a = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    #loss = compute_mse(y, tx, w)
    return w #, loss