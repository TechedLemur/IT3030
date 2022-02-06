import numpy as np


class Layer():

    def __init__(self, layerConfig=None) -> None:
        # init random weights between 0 and 1
        W = np.random.rand(layerConfig.inD, layerConfig.outD)
        # Scale initial weights
        self.W = W * \
            layerConfig.initial_weight_range[0] + \
            (1-W) * layerConfig.initial_weight_range[1]
        # init random bias weights between 0 and 1
        b = np.random.rand(layerConfig.outD)
        # Scale bias weights
        self.b = b * \
            layerConfig.initial_weight_range[0] + \
            (1-b) * layerConfig.initial_weight_range[1]
        self.lr = layerConfig.lr
        self.l1_alpha = layerConfig.l1_alpha
        self.l2_alpha = layerConfig.l2_alpha
        self.a = None  # summed outputs
        self.z = None  # f(a)
        self.activation = layerConfig.activation  # Activation function

    # Performs the forward pass given input y (row vector)
    def forward_pass(self, y) -> np.array:
        self.y = y
        self.a = y @ self.W + self.b
        self.z = self.activation.f(self.a)

        return self.z.copy()

    # Jlz is the jacobian matrix J L/Z , where L is the loss function and Z is the next layer
    def backward_pass(self, Jlz):
        self.df = self.activation.f_prime(self.a)  # df = JZSum-diagonal
        self.Jzy = np.einsum('ij,i->ij', self.W.T, self.df)  # Numerator form
        self.Jzw_hat = np.outer(self.y, self.df)
        self.Jly = Jlz @ self.Jzy
        self.Jlw = Jlz * self.Jzw_hat  # Weight derivative
        self.Jlb = Jlz * self.df  # Bias derivative

        return self.Jly.copy()

    # Update weights and biases
    def update_weights(self):
        self.W -= self.lr * self.Jlw + self.l1_alpha * \
            np.sign(self.W) + self.l2_alpha * self.W

        self.b -= self.lr * self.Jlb

    def __str__(self) -> str:
        return f"Layer - shape: {self.W.shape}, f(a): {self.z}"

    def __repr__(self):
        return f"Layer - shape: {self.W.shape}, f(a): {self.z}, activation: {self.activation}"
