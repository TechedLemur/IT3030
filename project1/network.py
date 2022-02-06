import numpy as np
from layer import Layer
from activation_functions import Softmax

"""
The network class keeps track of the layers, and calculates the loss.
The class implements fit() and predict() methods, as is conventionally the names used for these methods in ML.
"""


class Network():

    def __init__(self, networkConfig) -> None:
        self.layers = []
        # Create layers
        for c in networkConfig.layersConfig:
            self.layers.append(Layer(layerConfig=c))
        self.l1_alpha = networkConfig.l1_alpha  # l1 regularization constant
        self.l2_alpha = networkConfig.l2_alpha  # l2 regularization constant
        self.loss_function = networkConfig.loss_function  # loss function
        # Boolean value telling the network if Softmax should be applied to the last layer
        self.softmax = networkConfig.softmax

    # The fit method trains the network on the training set. Validation set can be added optionally.
    # This implementation does not support minibatching, performs the backward pass on one case at the time.
    def fit(self, x_train, y_train, epochs=100, valid=None, verbose=False):
        self.train_scores = []
        self.valid_scores = []
        self.omega1 = []
        self.omega2 = []
        for i in range(epochs):
            score = 0
            for x, y in zip(x_train, y_train):
                result = self.forward_pass(x)

                Jlz = self.loss_function.f_prime(y, result)

                # Apply softmax if enabled
                if self.softmax:
                    Jsz = Softmax.jacobian(result)
                    Jlz = Jlz @ Jsz

                # Calculate loss
                loss = self.loss_function.f(y, result)

                score += loss

                # Perform the backward pass through the layers
                self.backward_pass(Jlz)

                # Update the weights in the layers
                self.update_weights()

                if verbose:
                    print(
                        f"Input: {x} \nOutput: {result} \nTarget: {y} \nLoss: {loss}")

            # Save loss score
            self.train_scores.append(np.array([i, score / len(x_train)]))

            if valid:
                score = 0
                for x, y in zip(valid.x, valid.y):
                    score += self.loss_function.f(y, self.predict(x))

                # Save validation score
                self.valid_scores.append(np.array([i, score / (len(valid.x))]))

            o1 = 0
            o2 = 0
            # Calculate regularization costs
            for l in self.layers:
                o1 += np.sum(np.abs(l.W))
                o2 += 0.5 * np.sum(l.W ** 2)

            # Save regularization costs
            self.omega1.append(np.array([i, o1 * self.l1_alpha]))
            self.omega2.append(np.array([i, o2 * self.l2_alpha]))

        return self.train_scores, self.valid_scores

    # Make a prediction
    def predict(self, x_test):
        return self.forward_pass(x_test)

    # Perform forward pass
    def forward_pass(self, x):
        o = x
        for l in self.layers:
            o = l.forward_pass(o)

        if self.softmax:
            return Softmax.f(o)
        return o

    def update_weights(self):
        for l in self.layers:
            l.update_weights()

    def backward_pass(self, Jlz):
        jlz = Jlz
        for i in range(len(self.layers)-1, -1, -1):
            # Go backwards through the network and update weights
            jlz = self.layers[i].backward_pass(jlz)

    # Calculate test loss from test_set
    def test_loss(self, test_set):
        score = 0
        for x, y in zip(test_set.x, test_set.y):
            score += self.loss_function.f(y, self.predict(x))

        score = score / len(test_set.x)

        return score

    def __str__(self) -> str:
        return f"{self.layers} Softmax: {self.softmax} L1: {self.l1_alpha}, L2: {self.l2_alpha}"

    def __repr__(self):
        return f"{self.layers} Softmax: {self.softmax} L1: {self.l1_alpha}, L2: {self.l2_alpha}"
