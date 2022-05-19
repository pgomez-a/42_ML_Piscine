import matplotlib.pyplot as plt
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

def plot(x, y, theta):
    """
    Plot the data and prediction line from three non-empty numpy.array.
    """
    if not type(x) == np.ndarray or not type(y) == np.ndarray or not type(theta) == np.ndarray:
        return
    if not type(x) == np.ndarray or x.size == 0 or x.size != x.shape[0]:
        return
    if not type(y) == np.ndarray or y.size == 0 or y.size != y.shape[0] or x.size != y.size:
        return
    if not type(theta) == np.ndarray or theta.size != 2 or theta.size != theta.shape[0]:
        return
    y_ = predict_(x,theta)
    plt.scatter(x,y)
    plt.plot(x,y_)
    plt.show()
    return

def loss_elem_(y, y_hat):
    """
    Calculates all the elements (y_pred - y)^2 of the loss function.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size == 0:
        return
    if y.size != y_hat.size:
        return
    y = y.reshape(1, len(y))
    y_hat = y_hat.reshape(1, len(y_hat))
    return (y_hat - y)**2

def loss_(y, y_hat):
    """
    Calculates the value of the loss function.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size == 0:
        return
    if y.size != y_hat.size:
        return
    return np.sum(loss_elem_(y, y_hat)) / (2 * y.size)
