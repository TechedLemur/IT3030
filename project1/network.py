from copy import deepcopy
import numpy as np
from layer import Layer
from activation_functions import Sigmoid, ReLu, Softmax
from config import Globals


class Network():

    def __init__(self, layersConfig: list) -> None:
        self.layers = []
        for c in layersConfig:
            self.layers.append(Layer(layerConfig=c))
        self.l1_alpha = Globals.L1_ALPHA
        self.l2_alpha = Globals.L2_ALPHA
        self.loss_function = Globals.LOSS_FUNCION
        self.softmax = Globals.SOFTMAX

    def fit(self, x_train, y_train, epochs=Globals.EPOCHS, valid=None):
        self.train_scores = []
        self.valid_scores = []
        self.omega1 = []
        self.omega2 = []
        for i in range(epochs):
            score = 0
            for x, y in zip(x_train, y_train):
                result = self.forward_pass(x)  # Todo: Return omegas

                Jlz = self.loss_function.f_prime(y, result)
                # + self.l1_alpha * omega_1 +  self.l2_alpha * omega2

                if self.softmax:
                    Jsz = Softmax.jacobian(result)
                    Jlz = Jlz @ Jsz

                score += self.loss_function.f(y, result)

                self.backward_pass(Jlz)

            self.train_scores.append(np.array([i, score / len(x_train)]))

            if valid:
                score = 0
                for x, y in zip(valid.x, valid.y):
                    score += self.loss_function.f(y, self.predict(x))

                self.valid_scores.append(np.array([i, score / (len(valid.x))]))

            o1 = 0
            o2 = 0
            for l in self.layers:
                o1 += np.sum(np.abs(l.W))
                o2 += 0.5 * np.sum(l.W ** 2)

            self.omega1.append(np.array([i, o1 * self.l1_alpha]))
            self.omega2.append(np.array([i, o2 * self.l2_alpha]))

        return self.train_scores, self.valid_scores

    def predict(self, x_test):
        return self.forward_pass(x_test)

    def forward_pass(self, x):
        o = x
        for l in self.layers:
            o = l.forward_pass(o)

        if self.softmax:
            return Softmax.f(o)
        return o

    def backward_pass(self, Jlz):
        jlz = Jlz
        for i in range(len(self.layers)-1, -1, -1):
            # Go backwards through the network and update weights
            jlz = self.layers[i].backward_pass(jlz)

    def __str__(self) -> str:
        return f"{self.layers} Softmax: {self.softmax} L1: {self.l1_alpha}, L2: {self.l2_alpha}"

    def __repr__(self):
        return f"{self.layers} Softmax: {self.softmax} L1: {self.l1_alpha}, L2: {self.l2_alpha}"
