from importlib_metadata import SelectableGroups
import numpy as np
from activation_functions import sigmoid, d_sigmoid


class Layer():

    def __init__(self, inD, outD, lr=0.01, f=sigmoid, f_prime=d_sigmoid) -> None:
        # init random weights between -0.1 and 0.1
        self.W = (np.random.rand(inD, outD) - 0.5) / 5
        # init random bias weights between -0.1 and 0.1
        self.b = (np.random.rand(outD)-0.5) * 5
        self.lr = lr
        self.a = None  # summed outputs
        self.z = None  # f(a)
        self.f = f
        self.f_prime = f_prime
    # Performs the forward pass given input x (row vector)

    def forward_pass(self, y) -> np.array:
        self.a = y @ self.W + self.b
        self.z = self.f(self.a)
        self.df = self.f_prime(self.a)  # df = JZSum-diagonal
        self.Jzy = np.einsum('ij,i->ij', self.W.T, self.df)  # Numerator form
        self.Jzw_hat = np.outer(y, self.df)

        return self.z

    # Jlz is the jacobian matrix J L/Z , where L is the loss function and Z is the next layer
    def backward_pass(self, Jlz):
        self.Jly = Jlz @ self.Jzy
        self.Jlw = Jlz * self.Jzw_hat
        self.Jlb = Jlz * self.df

        self.W -= self.lr * self.Jlw
        self.b -= self.lr * self.Jlb

        return self.Jly

    def __str__(self) -> str:
        return f"Layer - shape: {self.W.shape}, f(a): {self.z}"

    def __repr__(self):
        return f"Layer - shape: {self.W.shape}, f(a): {self.z}"
