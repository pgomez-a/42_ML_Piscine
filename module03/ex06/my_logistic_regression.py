import numpy as np
import math

class MyLogisticRegression(object):
    """
    My personal logistic regression to classify things.
    """
    def __init__(self, theta, alpha = 0.001, max_iter = 100000, eps = 1e-15):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.eps = eps
        return

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
        x = np.insert(x, 0, [1] * x.shape[0], 1)
        return self.sigmoid_(np.matmul(x, self.theta))

    def loss_elem_(y, y_hat, eps = 1e-15):
        """
        Computes the logistic loss value.
        """
        if not type(y) == int and not type(y) == float: 
            return
        if not type(y_hat) == int and not type(y_hat) == float:
            return
        if not type(self.eps) == float:
            return
        y_hat += self.eps
        return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

    def loss_(self, y, y_hat):
        """
        Computes the logistic loss value.
        """
        if not type(y) == np.ndarray or y.size == 0:
            return
        if not type(y_hat) == np.ndarray or y_hat.size != y.size:
            return
        if not type(self.eps) == float:
            return
        return -sum(y * np.log(y_hat + self.eps) + (1 - y) * np.log((1 - y_hat) + self.eps)) / y.size

    def fit_(self, x, y):
        """
        Fits the model using gradient descent.
        """
        try:
            for i in range(self.max_iter):
                tmp_theta = self.gradient_(x, y, self.theta)
                self.theta -= self.alpha * tmp_theta
            return self.theta
        except:
            return


    @staticmethod
    def sigmoid_(x):
        """
        Computes the sigmoid of a vector.
        """
        if not type(x) == np.ndarray or x.size == 0:
            return
        return 1 / (1 + math.e**(-x))

    @staticmethod
    def gradient_(x, y, theta):
        """
        Computes a gradient vector from three non-empty numpy.ndarray.
        """
        if not type(x) == np.ndarray or not type(y) == np.ndarray:
            return
        if x.size == 0 or y.size == 0:
            return
        if not type(theta) == np.ndarray or theta.size == 0:
            return
        try:
            y_hat = MyLogisticRegression(theta).predict_(x)
            x = np.insert(x, 0, [1] * x.shape[0], 1)
            output = np.matmul(x.transpose(), y_hat - y) / y.size
            return output
        except:
             return
