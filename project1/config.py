from ast import Global
from msilib.schema import Class
from activation_functions import ReLu, Sigmoid, Softmax
from loss_functions import MSE


class Globals:
    LOSS_FUNCION = MSE  # Options: MSE, CROSS_ENTROPY
    LR = 0.01
    L1_ALPHA = 0.000001
    L2_ALPHA = 0.000001
    SOFTMAX = False


class LayerConfig:

    def __init__(self, inD=None, outD=None, lr=Globals.LR, activation=ReLu) -> None:
        self.inD = inD
        self.outD = outD
        self.lr = lr
        self.activation = activation


class Config:

    """
    Adding Layers:
    For each layer, specify number of input and output neurons. 
    Other parameters:
    Learning rate: lr (default = 0.01)
    Activation function f (default = Relu)
    Regularization parameters

    Please ensure that output dimension of layer N is equal to input dimension of layer N+1

    """

    layers = [


        LayerConfig(inD=30,  # Input dimension
                    outD=10,  # Output dimension
                    lr=0.01,  # Learning rate
                    activation=ReLu,  # Activation function
                    ),

        LayerConfig(inD=10,  # Input dimension
                    outD=1,  # Output dimension
                    lr=0.01,  # Learning rate
                    activation=Sigmoid,  # Activation function
                    ),


    ]

    Dataset = []
