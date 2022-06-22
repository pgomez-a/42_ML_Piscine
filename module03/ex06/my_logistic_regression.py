import numpy as np
import math
import sys

class MyLogisticRegression(object):
    """
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha = 0.001, max_iter = 1000, eps = 1e-15):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta.reshape((-1, 1))
        self.eps = eps
        return

    def sigmoid_(self, x):
        """
        Computes the sigmoid of a vector.
        """
        if not type(x) == np.ndarray or x.size == 0:
            return
        return 1 / (1 + math.e ** -x)

    def predict_(self, x):
        """
        Computes the vector of prediction y_hat from two non-emtpy
        numpy.ndarray.
        """
        if not type(x) == np.ndarray or x.size == 0:
            return
        if not type(self.theta) == np.ndarray or self.theta.size == 0:
            return
        if x.shape[1] != self.theta.shape[0] - 1:
            return
        x = np.insert(x, 0, 1, 1)
        return self.sigmoid_(np.matmul(x, self.theta)).reshape((-1, 1)).astype(float)

    def loss_(self, y_hat, y):
        """
        Computes the logistic loss value.
        """
        if not type(y) == np.ndarray or y.size == 0:
            return
        if not type(y_hat) == np.ndarray or y_hat.size != y.size:
            return
        if not type(self.eps) == float:
            return
        y_hat[y_hat == 0.] += self.eps
        y_hat[y_hat == 1.] -= self.eps
        return -sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / y.size

    def log_gradient_(self, x, y):
        """
        Computes a gradient vector from three non-empty numpy.ndarray.
        """
        if not type(x) == np.ndarray or not type(y) == np.ndarray:
            return
        if x.size == 0 or y.size == 0:
            return
        y_hat = self.predict_(x)
        x = np.insert(x, 0, 1, 1)
        return np.matmul(x.transpose(), y_hat - y) / y.size

    def fit_(self, x, y):
        """
        Computes gradient descent and fits the model.
        """
        for i in range(self.max_iter):
            self.theta -= (self.alpha * self.log_gradient_(x, y))
        return self.theta
