import numpy as np
import math

def sigmoid_(x):
    """
    Computes the sigmoid of a vector.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return
    return 1 / (1 + math.e**(-x))

def logistic_predict_(x, theta):
    """
    Computes the vector of prediction y_hat from two non-emtpy
    numpy.ndarray.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return
    if not type(theta) == np.ndarray or theta.size == 0:
        return
    if x.shape[1] != theta.shape[0] - 1:
        return
    x = np.insert(x, 0, [1] * x.shape[0], 1)
    return sigmoid_(np.matmul(x, theta))

def vec_log_loss_(y, y_hat, eps = 1e-15):
    """
    Computes the logistic loss value.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size != y.size:
        return
    if not type(eps) == float:
        return
    y_hat += eps
    return -sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / y.size
