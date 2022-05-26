import numpy as np

def simple_predict(x, theta):
    """
    Computes the prediction vector y_hat from two non-empy
    numpy.array.
    """
    if not type(x) == np.ndarray or not type(theta) == np.ndarray:
        return
    if x.size == 0 or theta.size == 0:
        return
    if x.size / len(x) + 1 != len(theta):
        return
    x = np.insert(x, 0, [1] * len(x), 1)
    return np.matmul(x.astype(float), theta.astype(float))
