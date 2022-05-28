import numpy as np

def gradient(x, y, theta):
    """
    Computes the gradient vector from three non-empty numpy.array,
    without any for-loop.
    """
    if not type(x) == np.ndarray or not type(y) == np.ndarray or not type(theta) == np.ndarray:
        return
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return
    if len(x) != len(y) or len(theta[0]) != len(y[0]) or len(x[0]) != len(theta) - 1:
        return
    x = np.insert(x, 0, [1] * len(x), 1)
    y_hat = np.matmul(x.astype(float), theta.astype(float))
    return np.matmul(x.transpose(), (y_hat - y).astype(float)) / len(x)
