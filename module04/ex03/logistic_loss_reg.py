import numpy as np

def l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray,
    without any for-loop.
    """
    theta[0][0] = 0.
    return np.matmul(theta.transpose(), theta).astype(float)[0][0]

def reg_log_loss_(y, y_hat, theta, lambda_):
    """
    Computes the regularized loss of a logistic regression model from two
    non-empty numpy.array, without any for loop.
    """
    if not type(y) == np.ndarray or y.size == 0:
        return
    if not type(y_hat) == np.ndarray or y_hat.size == 0:
        return
    if y.shape != y_hat.shape:
        return
    cost = sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return -(cost / y.size) + ((lambda_ * l2(theta)) / (2 * y.size))
