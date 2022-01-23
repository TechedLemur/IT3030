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
