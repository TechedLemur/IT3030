import numpy as np


def d_MSE(y_true, y_pred):
    return y_pred-y_true


def MSE(y_true, y_pred):
    return np.sum(0.5*(y_true-y_pred)**2)
