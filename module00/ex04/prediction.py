from copy import deepcopy
import numpy as np

def add_intercept(x):
    """
    Adds a column of 1's to the non-empty numpy.array x.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return None
    if x.size == x.shape[0]:
        x = x.reshape(len(x), 1)
    return np.insert(x, 0, [1] * x.shape[0], 1)

def predict_(x, theta):
    """
    Computes the vector of prediction y_hat from two non-empty numpy.array.
    """
    if not type(x) == np.ndarray or x.size == 0 or x.size != x.shape[0]:
        return None
    if not type(theta) == np.ndarray or theta.size != 2 or theta.size != theta.shape[0]:
        return None
    x = add_intercept(x)
    theta = theta.reshape(1, len(theta))
    return np.matmul(theta, x.transpose())[0]
