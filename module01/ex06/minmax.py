import numpy as np

def minmax(x):
    """
    Computes the normalized version of a non-empty numpy.ndarray using the min-max
    standardization.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return
    min_val = min(x)
    max_val = max(x)
    return (x - min_val) / (max_val - min_val)
