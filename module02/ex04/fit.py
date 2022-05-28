import numpy as np

def predict_(x, theta):
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

def fit_(x, y, theta, alpha, max_iter):
    """
    Fits the model to the training dataset contained in x and y.
    """
    if not type(x) == np.ndarray or not type(y) == np.ndarray or not type(theta) == np.ndarray:
        return
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return
    if len(x) != len(y) or len(theta[0]) != len(y[0]) or len(x[0]) != len(theta) - 1:
        return
    if not type(alpha) == float or not type(max_iter) == int:
        return
    for i in range(max_iter):
        tmp_theta = gradient(x, y, theta)
        theta -= alpha * tmp_theta
    return theta
