import numpy as np

def loss_(y, y_hat):
    """
    Computes the mean squared error of two non-empty numpy.array, without
    any for loop.
    """
    if not type(y) == np.ndarray or not type(y_hat) == np.ndarray:
        return
    if y.size == 0 or y_hat.size == 0:
        return
    if y.size != y_hat.size or len(y) != len(y_hat):
        return
    return (sum((y_hat - y)**2) / (2 * len(y)))[0]
