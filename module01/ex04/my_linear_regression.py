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

    def add_intercept_(self, x):
        """
        Adds a column of 1's to the non-empty numpy.array x.
        """
        if not type(x) == np.ndarray or x.size == 0:
            return None
        if x.size == x.shape[0]:
            x = x.reshape(len(x), 1)
        return np.insert(x, 0, [1] * x.shape[0], 1)

    def predict_(self, x):
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        """
        if not type(x) == np.ndarray or x.size == 0:
            return None
        x = self.add_intercept_(x)
        thetas = self.thetas.reshape(1, len(self.thetas))
        return np.matmul(thetas, x.transpose())[0]

    def gradient_(self, x, y):
        """
        Computes a gradient vector from three non-empty numpy.array,
        without any for loop.
        """
        if not type(x) == np.ndarray or x.size == 0:
            return
        if not type(y) == np.ndarray or y.size == 0:
            return
        y_hat = self.predict_(x).reshape(-1, 1)
        theta0_deriv = np.sum(y_hat - y) / y.size
        theta1_deriv = np.sum((y_hat - y) * x) / y.size
        return np.array([theta0_deriv, theta1_deriv]).reshape((-1, 1))

    def fit_(self, x, y):
        """
        Fits the model to the training dataset contained in x and y.
        """
        if not type(x) == np.ndarray or x.size == 0:
            return
        if not type(y) == np.ndarray or y.size == 0:
            return
        for i in range(self.max_iter):
            tmpTheta = self.gradient_(x, y)
            self.thetas[0][0] = self.thetas[0][0] - self.alpha * tmpTheta[0][0]
            self.thetas[1][0] = self.thetas[1][0] - self.alpha * tmpTheta[1][0]
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
        return (y_hat - y)**2

    @staticmethod
    def mse_(y, y_hat):
        """
        Calculates the value of the loss function.
        """
        if not type(y) == np.ndarray or y.size == 0:
            return
        if not type(y_hat) == np.ndarray or y_hat.size == 0:
            return
        if y.size != y_hat.size:
            return
        return np.sum(MyLinearRegression.loss_elem_(y, y_hat)) / y.size
