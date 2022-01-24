from mimetypes import init
import numpy as np

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

    def f(self, x):
        return sigmoid(x)

    def f_prime(self, x):
        return d_sigmoid(x)

    def __str__(self) -> str:
        return "Sigmoid"

    def __repr__(self):
        return "Sigmoid"


class Softmax():

    def f(self, x):
        return softmax(x)

    def jacobian(self, s):
        return d_softmax(s)

    def f_prime(self, x):
        s = softmax(x)
        return d_softmax(s)

    def __str__(self) -> str:
        return "Softmax"

    def __repr__(self):
        return "Softmax"


class ReLu():
    def f(self, x):
        return np.maximum(0, x)

    def f_prime(self, x):
        return np.greater(x, 0).astype(np.float32)

    def __str__(self) -> str:
        return "ReLu"

    def __repr__(self):
        return "ReLu"
