import numpy as np

def accuracy_score_(y, y_hat):
    """
    Computes the accuracy score.
    """
    try:
        true = 0
        for pos in range(len(y)):
            if y[pos] == y_hat[pos]:
                true += 1
        return true / y.size
    except:
        return

def precision_score_(y, y_hat, pos_label = 1):
    """
    Computes the precision score.
    """
    try:
        true_pos = 0
        false_pos = 0
        for pos in range(len(y)):
            if y[pos] == y_hat[pos] and y_hat[pos] == pos_label:
                true_pos += 1
            if y[pos] != y_hat[pos] and y_hat[pos] == pos_label:
                false_pos += 1
        output = true_pos / (true_pos + false_pos)
        return output
    except:
        return

def recall_score_(y, y_hat, pos_label = 1):
    """
    Computes the recall score.
    """
    try:
        true_pos = 0
        false_neg = 0
        for pos in range(len(y)):
            if y[pos] == y_hat[pos] and y_hat[pos] == pos_label:
                true_pos += 1
            if y[pos] != y_hat[pos] and y_hat[pos] != pos_label:
                false_neg += 1
        output = true_pos / (true_pos + false_neg)
        return output
    except:
        return

def f1_score_(y, y_hat, pos_label = 1):
    """
    Computes the f1 score.
    """
    try:
        precision = precision_score_(y, y_hat, pos_label)
        recall = recall_score_(y, y_hat, pos_label)
        output = (2 * precision * recall) / (precision + recall)
        return output
    except:
        return
