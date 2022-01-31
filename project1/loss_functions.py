import numpy as np


class MSE:

    @staticmethod
    def f_prime(y_true, y_pred):
        return y_pred-y_true

    @staticmethod
    def f(y_true, y_pred):
        return np.sum(0.5*(y_true-y_pred)**2)
