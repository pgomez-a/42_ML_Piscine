import numpy as np
import math

def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return
    return 1 / (1 + math.e**-x)
