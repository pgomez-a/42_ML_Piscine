import numpy as np
import math

def zscore(x):
    """
    Computes the normalized version of a non-empy numpy.ndarray using the z-score standarization.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return
    mean = sum(x) / x.size
    std_dev = math.sqrt(sum((x - mean)**2) / x.size)
    return (x - mean) / std_dev
