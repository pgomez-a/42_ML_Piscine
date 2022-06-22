import numpy as np

def add_polynomial_features(x, power):
    """
    Add polyonmial features to matrix x by raising its columns to
    every power in the range of 1 up to the power given.
    """
    if not type(x) == np.ndarray or x.size == 0 or power <= 0:
        return
    row_size = x.shape[1]
    for y in range(2, power + 1):
        for t in range(row_size):
            app = (x[:, t] ** y)
            app = app.reshape(app.shape[0], 1)
            x = np.append(x, app, 1)
    return x
