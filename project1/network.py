import numpy as np
from layer import Layer
from activation_functions import softmax


class Network():

    def __init__(self, neurons: np.array, lr=0.01) -> None:
        self.layers = []
        for i in range(len(neurons)-1):
            self.layers.append(Layer(neurons[i], neurons[i+1]))
        self.lr = lr

    def fit(self, x_train, y_train, epochs=100):
        scores = []
        for i in range(epochs):
            score = 0
            for x, y in zip(x_train, y_train):
                result = self.forward_pass(x)

                Jlz = self.d_l2_loss(y, result)
                score += self.l2_loss(y, result)

                self.backward_pass(Jlz)

            scores.append(np.array([i, score / len(x_train)]))

        return scores

    def d_l2_loss(self, y_true, y_pred):
        return y_pred-y_true

    def l2_loss(self, y_true, y_pred):
        return np.sum(0.5*(y_true-y_pred)**2)

    def predict(self, x_test):
        return self.forward_pass(x_test)

    def forward_pass(self, x):
        o = x
        for l in self.layers:
            o = l.forward_pass(o)
        return o
        # return softmax(o)

    def backward_pass(self, Jlz):
        jlz = Jlz
        for i in range(len(self.layers)-1, -1, -1):
            # Go backwards through the network and update weights
            jlz = self.layers[i].backward_pass(jlz)

    def __str__(self) -> str:
        return f"{self.layers}"

    def __repr__(self):
        return f"{self.layers}"
