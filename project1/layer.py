from importlib_metadata import SelectableGroups
import numpy as np
from activation_functions import sigmoid, d_sigmoid, Sigmoid, ReLu
from config import Globals


class Layer():

    def __init__(self, layerConfig=None) -> None:
        # init random weights between -0.1 and 0.1
        self.W = (np.random.rand(layerConfig.inD, layerConfig.outD) - 0.5) / 5
        # init random bias weights between -0.1 and 0.1
        self.b = (np.random.rand(layerConfig.outD)-0.5) * 5
        self.lr = layerConfig.lr
        self.l1_alpha = Globals.L1_ALPHA
        self.l2_alpha = Globals.L2_ALPHA
        self.a = None  # summed outputs
        self.z = None  # f(a)
        self.activation = layerConfig.activation

    # Performs the forward pass given input x (row vector)
    def forward_pass(self, y) -> np.array:
        self.a = y @ self.W + self.b
        self.z = self.activation.f(self.a)
        self.df = self.activation.f_prime(self.a)  # df = JZSum-diagonal
        self.Jzy = np.einsum('ij,i->ij', self.W.T, self.df)  # Numerator form
        self.Jzw_hat = np.outer(y, self.df)

        return self.z.copy()

    # Jlz is the jacobian matrix J L/Z , where L is the loss function and Z is the next layer
    def backward_pass(self, Jlz):
        self.Jly = Jlz @ self.Jzy
        self.Jlw = Jlz * self.Jzw_hat
        self.Jlb = Jlz * self.df

        self.W -= self.lr * self.Jlw + self.l1_alpha * \
            np.sign(self.W) + self.l2_alpha * self.W

        self.b -= self.lr * self.Jlb

        return self.Jly.copy()

    def __str__(self) -> str:
        return f"Layer - shape: {self.W.shape}, f(a): {self.z}"

    def __repr__(self):
        return f"Layer - shape: {self.W.shape}, f(a): {self.z}, activation: {self.activation}"
