import numpy as np

class MyLinearRegression(object):
    """
    My personal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha = 0.001, max_iter = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape(-1, 1)
        return

    def predict_(self, x):
        """
        Computes the prediction vector y_hat from two non-empy
        numpy.array.
        """
        if not type(x) == np.ndarray:
            return
        if x.size == 0:
            return
        if x.size / len(x) + 1 != len(self.thetas):
            return
        x = np.insert(x, 0, [1] * len(x), 1)
        return np.matmul(x.astype(float), self.thetas.astype(float))

    @staticmethod
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

    def fit_(self, x, y):
        """
        Fits the model to the training dataset contained in x and y.
        """
        if not type(x) == np.ndarray or not type(y) == np.ndarray:
            return
        if x.size == 0 or y.size == 0:
            return
        if len(x) != len(y) or len(self.thetas[0]) != len(y[0]) or len(x[0]) != len(self.thetas) - 1:
            return
        for i in range(self.max_iter):
            tmp_theta = self.gradient(x, y, self.thetas)
            self.thetas -= self.alpha * tmp_theta
        return self.thetas

    @staticmethod
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
        return ((y_hat - y)**2).reshape((-1, 1))

    @staticmethod
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
        return np.sum(MyLinearRegression.loss_elem_(y, y_hat)) / (2 * y.size)
