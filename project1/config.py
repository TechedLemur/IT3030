from ast import Global
from calendar import EPOCH
from msilib.schema import Class
from activation_functions import ReLu, Sigmoid, Softmax
from loss_functions import MSE, CrossEntropy


class Globals:
    LOSS_FUNCION = MSE  # Options: MSE, CrossEntropy
    LR = 0.01
    L1_ALPHA = 0.000000
    L2_ALPHA = 0.000000
    SOFTMAX = False

    EPOCHS = 100


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


        LayerConfig(inD=256,  # Input dimension
                    outD=100,  # Output dimension
                    lr=0.01,  # Learning rate
                    activation=ReLu,  # Activation function
                    ),

        LayerConfig(inD=100,  # Input dimension
                    outD=50,  # Output dimension
                    lr=0.01,  # Learning rate
                    activation=ReLu,  # Activation function
                    ),

        LayerConfig(inD=50,  # Input dimension
                    outD=4,  # Output dimension
                    lr=0.01,  # Learning rate
                    activation=Sigmoid,  # Activation function
                    ),

        # LayerConfig(inD=10,  # Input dimension
        #             outD=1,  # Output dimension
        #             lr=0.01,  # Learning rate
        #             activation=Sigmoid,  # Activation function
        #             ),


    ]

    # Dataset parameters
    # Size of the whole dataset, including validation and test set.
    DATASET_SIZE = 500

    NOISE_PROBABILITY = 0.01

    IMAGE_SIZE = 16

    # Position bundaries for the center of cross/circle, and the position of the horizontal/vertical bars
    X_RANGE = (4, 12)
    Y_RANGE = (4, 12)

    RADIUS_RANGE = (4, 7)  # Radius range for circle and cross

    TEST_SET_PORTION = 0.1
    VALIDATION_SET_PORTION = 0.1

    # Training set becomes the remaining portion
