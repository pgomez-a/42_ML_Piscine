import pandas as pd
import numpy as np

def get_labels(y_true, y_hat, valid_labels):
    """
    Gets the valid labels for the given values.
    """
    y_values = sorted(set(np.append(y_true, y_hat)))
    labels = dict()
    pos = 0
    for label in y_values:
        if valid_labels == None or (valid_labels != None and label in valid_labels):
            labels[label] = pos
            pos += 1
    return labels

def confusion_matrix_(y_true, y_hat, labels = None, df_option = False):
    """
    Computes confusion matrix to evaluate the accuracy of a classification.
    """
    valid_labels = get_labels(y_true, y_hat, labels)
    matrix = np.zeros([len(valid_labels), len(valid_labels)])
    for label in range(len(y_true)):
        if labels == None:
            matrix[valid_labels[y_true[label][0]]][valid_labels[y_hat[label][0]]] += 1
        elif y_true[label][0] in labels and y_hat[label][0] in labels:
            matrix[valid_labels[y_true[label][0]]][valid_labels[y_hat[label][0]]] += 1
    if df_option == True:
        return pd.DataFrame(matrix.astype(int), columns = valid_labels, index = valid_labels)
    return matrix.astype(int)
