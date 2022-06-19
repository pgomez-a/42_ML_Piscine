import numpy as np

def accuracy_score_(y, y_hat):
    """
    Computes the accuracy score.
    """
    if not type(y) == np.ndarray or not type(y_hat) == np.ndarray:
        return
    if y.shape != y_hat.shape:
        return
    true_value = 0
    for pos in range(len(y)):
        if y[pos] == y_hat[pos]:
            true_value += 1
    return true_value / y.size


def precision_score_(y, y_hat, pos_label = 1):
    """
    Computes the precision score.
    """
    if not type(y) == np.ndarray or not type(y_hat) == np.ndarray:
        return
    if y.shape != y_hat.shape:
        return
    if not type(pos_label) == int and not type(pos_label) == str:
        return
    true_positive = 0
    false_positive = 0
    for pos in range(len(y)):
        if y[pos] == pos_label and y_hat[pos] == pos_label:
            true_positive += 1
        if y[pos] != pos_label and y_hat[pos] == pos_label:
            false_positive += 1
    return true_positive / (true_positive + false_positive)

def recall_score_(y, y_hat, pos_label = 1):
    """
    Computes the recall score.
    """
    if not type(y) == np.ndarray or not type(y_hat) == np.ndarray:
        return
    if y.shape != y_hat.shape:
        return
    if not type(pos_label) == int and not type(pos_label) == str:
        return
    true_positive = 0
    false_negative = 0
    for pos in range(len(y)):
        if y[pos] == pos_label and y_hat[pos] == pos_label:
            true_positive += 1
        if y[pos] == pos_label and y_hat[pos] != pos_label:
            false_negative += 1
    return true_positive / (true_positive + false_negative)

def f1_score_(y, y_hat, pos_label = 1):
    """
    Compute the f1 score.
    """
    if not type(y) == np.ndarray or not type(y_hat) == np.ndarray:
        return
    if y.shape != y_hat.shape:
        return
    if not type(pos_label) == int and not type(pos_label) == str:
        return
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return 2 * (precision * recall) / (precision + recall)
