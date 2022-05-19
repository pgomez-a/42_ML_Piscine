import numpy as np

def loss_(y, y_hat):
    """
    Computes the half mean squared error of two non-empty numpy.array,
    without any for loop.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size == 0:
        return
    if y.size != y_hat.size:
        return
    return np.sum((y - y_hat) * (y - y_hat)) / (2 * y.size)
