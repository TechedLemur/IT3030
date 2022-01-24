import numpy as np
from layer import Layer
from activation_functions import Sigmoid, ReLu
from loss_functions import MSE, d_MSE


class Network():

    def __init__(self, neurons: np.array, lr=0.01, loss=MSE, d_loss=d_MSE) -> None:
        self.layers = []
        for i in range(len(neurons)-1):
            if i == len(neurons)-2:
                self.layers.append(
                    Layer(neurons[i], neurons[i+1], lr=lr, activation=Sigmoid()))
            else:
                self.layers.append(Layer(neurons[i], neurons[i+1], lr=lr))
        self.lr = lr
        self.loss = MSE
        self.d_loss = d_MSE

    def fit(self, x_train, y_train, epochs=100):
        scores = []
        for i in range(epochs):
            score = 0
            for x, y in zip(x_train, y_train):
                result = self.forward_pass(x)

                Jlz = self.d_loss(y, result)
                score += self.loss(y, result)

                self.backward_pass(Jlz)

            scores.append(np.array([i, score / len(x_train)]))

        return scores

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
