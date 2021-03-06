import numpy as np

"""
This file contains classes for the different activation funcions. Each function has an f, and f_prime method.
"""


# Sigmoid activation function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative


def d_sigmoid(x):
    return (sigmoid(x) * (1-sigmoid(x)))

# Softmax activation function


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# Calculates the softmax Jacobian. Input s is the softmax output


def d_softmax(s):
    r = np.outer(-s, s)
    for i, a in enumerate(s):
        r[i, i] = a-a**2
    return r


class Sigmoid():
    @staticmethod
    def f(x):
        return sigmoid(x)

    @staticmethod
    def f_prime(x):
        return d_sigmoid(x)

    def __str__(self) -> str:
        return "Sigmoid"

    def __repr__(self):
        return "Sigmoid"


class Softmax():
    @staticmethod
    def f(x):
        return softmax(x)

    @staticmethod
    def jacobian(s):
        return d_softmax(s)

    @staticmethod
    def f_prime(x):
        s = softmax(x)
        return d_softmax(s)

    def __str__(self) -> str:
        return "Softmax"

    def __repr__(self):
        return "Softmax"


class ReLu():
    @staticmethod
    def f(x):
        return np.maximum(0, x)

    @staticmethod
    def f_prime(x):
        return np.greater(x, 0).astype(np.float32)

    def __str__(self) -> str:
        return "ReLu"

    def __repr__(self):
        return "ReLu"


class TanH():

    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def f_prime(x):
        f = TanH.f(x)
        return 1 - f ** 2

    def __str__(self) -> str:
        return "tanh"

    def __repr__(self):
        return "tanh"


class Linear():

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def f_prime(x):
        return np.ones(x.shape)
