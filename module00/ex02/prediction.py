import numpy as np

def simple_predict(x, theta):
    """
    Computes the vector of precition y_hat from two non-empty numpy.ndarray.
    """
    if not type(x) == np.ndarray or not type(theta) == np.ndarray:
        return None
    if x.ndim != 1 or theta.ndim != 1:
        return None
    if x.size <= 2 or theta.size != 2:
        return None
    predict = theta[0] + theta[1] * x
    return predict
