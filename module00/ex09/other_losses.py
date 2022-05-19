import numpy as np
import math

def mse_(y, y_hat):
    """
    Calculate the MSE between the predicted output and the real output.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size == 0:
        return
    if y.size != y_hat.size:
        return
    return np.sum((y_hat - y)**2) / y.size

def rmse_(y, y_hat):
    """
    Calculate the RMSE between the predicted output and the real output.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size == 0:
        return
    if y.size != y_hat.size:
        return
    return math.sqrt(mse_(y, y_hat))

def mae_(y, y_hat):
    """
    Calculate the MAE between the predicted output and the real output.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size == 0:
        return
    if y.size != y_hat.size:
        return
    return np.sum(abs(y_hat - y)) / y.size

def r2score_(y, y_hat):
    """
    Calculate the R2score between the predicted output and the output.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size == 0:
        return
    if y.size != y_hat.size:
        return
    mean = np.sum(y) / y.size
    mean = np.array([mean] * y.size)
    return 1 - (mse_(y, y_hat) / mse_(y_hat, mean))
