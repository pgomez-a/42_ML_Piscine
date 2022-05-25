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
    if not type(x) == np.ndarray or x.size == 0:
        return None
    if not type(theta) == np.ndarray:
        return None
    x = add_intercept(x)
    theta = theta.reshape(1, len(theta))
    return np.matmul(theta, x.transpose())[0]

def simple_gradient(x, y , theta):
    """
    Computes a gradient vector from three non-empty numpy.array,
    without any for loop.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(theta) == np.ndarray or theta.size == 0:
        return
    y_hat = predict_(x, theta).reshape(-1, 1)
    theta0_deriv = np.sum(y_hat - y) / y.size
    theta1_deriv = np.sum((y_hat - y) * x) / y.size
    return np.array([theta0_deriv, theta1_deriv]).reshape((-1, 1))
