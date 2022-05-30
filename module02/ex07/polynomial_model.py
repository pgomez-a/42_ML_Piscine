import numpy as np

def add_polynomial_features(x, power):
    """
    Add polynomial features to vector x by raising its values up
    to the power given in argument.
    """
    if not type(x) == np.ndarray or x.size == 0:
        return
    if not type(power) == int:
        return
    output = x.reshape((-1, 1))
    for i in range(2, power + 1):
        new_power = x ** i
        output = np.insert(output, i - 1, new_power.transpose()[0], 1)
    return output
