from activation_functions import ReLu, Sigmoid, Linear, TanH
from loss_functions import MSE, CrossEntropy
import configparser
from data_generator import DataGenerator

"""
This is a utility file, hiding the details of parsing the config file.
"""


activation_functions = {'ReLu': ReLu,
                        'Sigmoid': Sigmoid,
                        'Linear': Linear,
                        'TanH': TanH}

loss_function = {'MSE': MSE, 'CrossEntropy':  CrossEntropy}


# Configuration class for a layer
class LayerConfig:

    def __init__(self, inD=None, outD=None, lr=0.01, activation=ReLu, initial_weight_range=(-0.5, 0.5), l1_alpha=0, l2_alpha=0) -> None:
        self.inD = inD
        self.outD = outD
        self.lr = lr
        self.activation = activation
        self.initial_weight_range = initial_weight_range
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha


# Configuration class for a network
class NetworkConfig:

    def __init__(self, layersConfig=None, loss_function=MSE, l1_alpha=0, l2_alpha=0, softmax=False) -> None:
        self.layersConfig = layersConfig
        self.loss_function = loss_function
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.softmax = softmax


# Returns a network configuration object from the given config file
def get_network_from_config_file(filename: str) -> (NetworkConfig):
    parser = configparser.ConfigParser()
    parser.read(filename)
    network_config = parser['network']
    loss = loss_function[network_config['loss_function']]
    l1 = float(network_config['l1_alpha'])
    l2 = float(network_config['l2_alpha'])
    softmax = bool(int(network_config['softmax']))

    layers_config = parser['layers']

    layers = []

    for l in layers_config.values():
        splitted = l.split(",")
        inD = int(splitted[0].strip())
        outD = int(splitted[1].strip())
        lr = float(splitted[2].strip())
        activation = activation_functions[splitted[3].strip()]
        rnge = splitted[4].split(":")
        rnge = (float(rnge[0].strip()), float(rnge[1].strip()))

        layers.append(LayerConfig(inD, outD, lr, activation, rnge, l1, l2))

    return NetworkConfig(layersConfig=layers, loss_function=loss, l1_alpha=l1, l2_alpha=l2, softmax=softmax)


# Returns datasets based on the given config file
def get_data_from_config_file(filename: str):
    parser = configparser.ConfigParser()
    parser.read(filename)

    conf = parser['dataset']

    dataset_size = int(conf['dataset_size'])
    t = float(conf['test_set_portion'])
    v = float(conf['validation_set_portion'])
    p = float(conf['noise_probability'])
    img_size = int(conf['image_size'])

    x_range = conf['x_range'].split(",")
    x_range = (int(x_range[0].strip()), int(x_range[1].strip()))

    y_range = conf['y_range'].split(",")
    y_range = (int(y_range[0].strip()), int(y_range[1].strip()))

    radius_range = conf['radius_range'].split(",")
    radius_range = int(radius_range[0].strip()), int(radius_range[1].strip())

    side_boundaries = conf['side_boundaries'].split(",")
    side_boundaries = int(side_boundaries[0].strip()), int(
        side_boundaries[1].strip())

    return DataGenerator.GetTrainTestValid(
        image_size=img_size,
        dataset_size=dataset_size,
        test_set_portion=t,
        valid_set_portion=v,
        noise=p,
        x_range=x_range,
        y_range=y_range,
        side_boundaries=side_boundaries,
        radius_range=radius_range)


# Get the run settings
def get_run_config(filename: str):
    parser = configparser.ConfigParser()
    parser.read(filename)
    return parser['run']
