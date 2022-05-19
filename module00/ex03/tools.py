from copy import deepcopy
import numpy as np

def add_intercept(x):
    """
    Adds a column of 1's to the non-empty numpy.array x.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return None
    try:
        bound = x.shape[1]
    except:
        x = x.reshape(len(x), 1)
    return np.insert(x, 0, [1] * x.shape[0], 1)
