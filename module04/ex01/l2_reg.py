import numpy as np

def iterative_l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray,
    with a for-loop.
    """
    output = 0.0
    for i in range(1, theta.shape[0]):
        output += theta[i][0] ** 2
    return output

def l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray,
    without any for-loop.
    """
    theta[0][0] = 0.
    return np.matmul(theta.transpose(), theta).astype(float)[0][0]
