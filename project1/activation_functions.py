import numpy as np

# Sigmoid activation function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative


def d_sigmoid(x):
    return (sigmoid(x) * (1-sigmoid(x)))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
