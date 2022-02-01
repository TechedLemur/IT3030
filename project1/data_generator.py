
from config import Config
import numpy as np


class Data():
    def __init__(self, img, y) -> None:
        self.y = y.astype(np.int8)
        self.img = img
        self.x = img.flatten().astype(np.int8)


class DataSet():
    def __init__(self, data_list) -> None:
        self.data = data_list
        self.x = list(map(lambda d: d.x, data_list))
        self.y = list(map(lambda d: d.y, data_list))


class DataGenerator:

    HORIZONTAL_ID = np.array([1, 0, 0, 0])
    VERTICAL_ID = np.array([0, 1, 0, 0])
    CROSS_ID = np.array([0, 0, 1, 0])
    CIRCLE_ID = np.array([0, 0, 0, 1])

    @staticmethod
    def HorizontalBar(size: int = 16, center=False):

        img = np.zeros((size, size), dtype=np.bool)

        if center:
            lower_bound = int(size/2) - 3
            upper_bound = int(size/2) + 4
        else:
            lower_bound = 1
            upper_bound = size - 1

        for i in np.random.choice(np.arange(lower_bound, upper_bound), np.random.randint(1, min(int(size/4), upper_bound-lower_bound - 1))):
            img[i, :] = np.True_
        return img

    @staticmethod
    def VerticalBar(size: int = 16, center=False):

        img = np.zeros((size, size), dtype=np.bool)

        if center:
            lower_bound = int(size/2) - 3
            upper_bound = int(size/2) + 4
        else:
            lower_bound = 1
            upper_bound = size - 1

        for i in np.random.choice(np.arange(lower_bound, upper_bound), np.random.randint(1, min(int(size/4), upper_bound-lower_bound - 1))):
            img[:, i] = np.True_
        return img

    @staticmethod
    def Cross(size: int = 16, center=False):

        img = np.zeros((size, size), dtype=np.bool)

        if center:
            lower_bound = int(size/2) - 1
            upper_bound = int(size/2) + 2
        else:
            lower_bound = 5
            upper_bound = size - 5

        x = np.random.randint(lower_bound, upper_bound)

        y = np.random.randint(x-2, x+3)

        h_length = min(x - 1, size - x-1,
                       np.random.random_integers(int(size/4), int(size/2)))
        v_length = min(y - 1, size - y-1,
                       np.random.random_integers(h_length-2, h_length+2))

        img[x-h_length:x+h_length, y] = np.True_
        img[x, y-v_length:y+v_length] = np.True_
        return img

    @staticmethod
    def Circle(size: int = 16, center=False):

        if center:
            lower_bound = int(size/2) - 1
            upper_bound = int(size/2) + 2
        else:
            lower_bound = 5
            upper_bound = size - 5

        x = np.random.randint(lower_bound, upper_bound)

        y = np.random.randint(x-2, x+3)

        r = min(x - 1, size - x-1, y - 1, size - y-1,
                np.random.random_integers(int(size/4), int(size/2)))

        x_ind, y_ind = np.indices((size, size))
        return np.abs(np.hypot(x - x_ind, y - y_ind)-r) < 0.5

    # Returns an image with added random noise with probability p

    @staticmethod
    def AddNoise(img, p=0.05):

        mask = np.random.choice(a=[True, False], size=img.shape, p=[p, 1-p])

        return np.bitwise_xor(img, mask)

    @staticmethod
    def GetCircles(n: int, size: int = 16, noise: float = 0.01, center=False):
        c = []

        for _ in range(n):
            c.append(Data(DataGenerator.AddNoise(
                DataGenerator.Circle(size, center)), DataGenerator.CIRCLE_ID))

        return c

    @staticmethod
    def GetHorizontalBars(n: int, size: int = 16, noise: float = 0.01, center=False):
        c = []

        for _ in range(n):
            c.append(Data(DataGenerator.AddNoise(DataGenerator.HorizontalBar(
                size, center)), DataGenerator.HORIZONTAL_ID))

        return c

    @staticmethod
    def GetVerticalBars(n: int, size: int = 16, noise: float = 0.01, center=False):
        c = []

        for _ in range(n):
            c.append(Data(DataGenerator.AddNoise(
                DataGenerator.VerticalBar(size, center)), DataGenerator.VERTICAL_ID))

        return c

    @staticmethod
    def GetCrosses(n: int, size: int = 16, noise: float = 0.01, center=False):
        c = []

        for _ in range(n):
            c.append(Data(DataGenerator.AddNoise(
                DataGenerator.Cross(size, center)), DataGenerator.CROSS_ID))

        return c

    @staticmethod
    def GetTrainTestValid():
        train_size = int(Config.DATASET_SIZE * (1 -
                         (Config.TEST_SET_PORTION + Config.VALIDATION_SET_PORTION)))

        if train_size < 0:
            raise Exception("Illegal test and train set portions")

        test_size = int((Config.DATASET_SIZE - train_size) * Config.TEST_SET_PORTION /
                        (Config.TEST_SET_PORTION + Config.VALIDATION_SET_PORTION))

        valid_size = Config.DATASET_SIZE - train_size - test_size

        h = int(0.25*train_size)
        v = int(0.25*train_size)
        cr = int(0.25*train_size)
        ci = train_size - (h+v+cr)

        train = DataGenerator.GetHorizontalBars(h, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + \
            DataGenerator.GetVerticalBars(
                v, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + DataGenerator.GetCrosses(
                cr, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + DataGenerator.GetCircles(
                ci, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES)

        h = int(0.25*test_size)
        v = int(0.25*test_size)
        cr = int(0.25*test_size)
        ci = test_size - (h+v+cr)

        test = DataGenerator.GetHorizontalBars(h, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + \
            DataGenerator.GetVerticalBars(
                v, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + DataGenerator.GetCrosses(
                cr, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + DataGenerator.GetCircles(
                ci, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES)

        h = int(0.25*valid_size)
        v = int(0.25*valid_size)
        cr = int(0.25*valid_size)
        ci = valid_size - (h+v+cr)

        valid = DataGenerator.GetHorizontalBars(h, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + \
            DataGenerator.GetVerticalBars(
                v, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + DataGenerator.GetCrosses(
                cr, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES) + DataGenerator.GetCircles(
                ci, Config.IMAGE_SIZE, Config.NOISE_PROBABILITY, Config.CENTER_IMAGES)

        np.random.shuffle(train)
        np.random.shuffle(test)
        np.random.shuffle(valid)
        return DataSet(train), DataSet(test), DataSet(valid)
