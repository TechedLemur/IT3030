import numpy as np
import matplotlib.pyplot as plt


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
    def HorizontalBar(size: int = 16, position_range=(1, 15), side_boundaries=(0, 16)):

        img = np.zeros((size, size), dtype=np.bool)

        lower_bound = max(1, position_range[0])
        upper_bound = min(position_range[1], size - 1)

        for i in np.random.choice(np.arange(lower_bound, upper_bound), np.random.randint(1, min(int(size/4), upper_bound-lower_bound - 1))):
            img[i, side_boundaries[0]:side_boundaries[1]] = np.True_
        return img

    @staticmethod
    def VerticalBar(size: int = 16, position_range=(1, 16), side_boundaries=(0, 16)):

        img = np.zeros((size, size), dtype=np.bool)

        lower_bound = max(1, position_range[0])
        upper_bound = min(position_range[1], size - 1)

        for i in np.random.choice(np.arange(lower_bound, upper_bound), np.random.randint(1, min(int(size/4), upper_bound-lower_bound - 1))):
            img[side_boundaries[0]:side_boundaries[1], i] = np.True_
        return img

    @staticmethod
    def Cross(size: int = 16, x_range=(1, 16), y_range=(1, 16), radius_range=(1, 15)):

        img = np.zeros((size, size), dtype=np.bool)

        lower_bound = max(5, x_range[0])
        upper_bound = min(x_range[1], size - 5)

        x = np.random.randint(lower_bound, upper_bound)

        lower_bound = max(5, y_range[0])
        upper_bound = min(y_range[1], size - 5)

        y = np.random.randint(lower_bound, upper_bound)

        h_length = min(x - 1, size - x-1,
                       np.random.random_integers(radius_range[0], radius_range[1]))
        v_length = min(y - 1, size - y-1,
                       np.random.random_integers(h_length-2, h_length+2))

        img[x-h_length:x+h_length, y] = np.True_
        img[x, y-v_length:y+v_length] = np.True_
        return img

    @staticmethod
    def Circle(size: int = 16, x_range=(1, 16), y_range=(1, 16), radius_range=(1, 15)):

        lower_bound = max(5, x_range[0])
        upper_bound = min(x_range[1], size - 5)

        x = np.random.randint(lower_bound, upper_bound)

        lower_bound = max(5, y_range[0])
        upper_bound = min(y_range[1], size - 5)

        y = np.random.randint(lower_bound, upper_bound)

        r = min(x - 1, size - x-1, y - 1, size - y-1,
                np.random.random_integers(radius_range[0], radius_range[1]))

        x_ind, y_ind = np.indices((size, size))
        return np.abs(np.hypot(x - x_ind, y - y_ind)-r) < 0.5

    # Returns an image with added random noise with probability p

    @staticmethod
    def AddNoise(img, p=0.05):

        mask = np.random.choice(a=[True, False], size=img.shape, p=[p, 1-p])

        return np.bitwise_xor(img, mask)

    @staticmethod
    def GetCircles(n: int, size: int = 16, noise: float = 0.01, x_range=(1, 16), y_range=(1, 16), radius_range=(1, 15)):
        c = []

        for _ in range(n):
            c.append(Data(DataGenerator.AddNoise(
                DataGenerator.Circle(size, x_range=x_range, y_range=y_range, radius_range=radius_range), noise), DataGenerator.CIRCLE_ID))

        return c

    @staticmethod
    def GetHorizontalBars(n: int, size: int = 16, noise: float = 0.01, position_range=(1, 16), side_boundaries=(0, 16)):
        c = []

        for _ in range(n):
            c.append(Data(DataGenerator.AddNoise(DataGenerator.HorizontalBar(
                size, position_range=position_range, side_boundaries=side_boundaries), noise), DataGenerator.HORIZONTAL_ID))

        return c

    @staticmethod
    def GetVerticalBars(n: int, size: int = 16, noise: float = 0.01, position_range=(1, 16), side_boundaries=(0, 16)):
        c = []

        for _ in range(n):
            c.append(Data(DataGenerator.AddNoise(
                DataGenerator.VerticalBar(size, position_range=position_range, side_boundaries=side_boundaries), noise), DataGenerator.VERTICAL_ID))

        return c

    @staticmethod
    def GetCrosses(n: int, size: int = 16, noise: float = 0.01, x_range=(1, 16), y_range=(1, 16), radius_range=(1, 15)):
        c = []

        for _ in range(n):
            c.append(Data(DataGenerator.AddNoise(
                DataGenerator.Cross(size, x_range=x_range, y_range=y_range, radius_range=radius_range), noise), DataGenerator.CROSS_ID))

        return c

    @staticmethod
    def GetTrainTestValid(
            image_size: int = 16,
            dataset_size: int = 1000,
            test_set_portion: float = 0.1,
            valid_set_portion: float = 0.1,
            noise: float = 0.01,
            x_range=(3, 13),
            y_range=(3, 13),
            side_boundaries=(0, 16),
            radius_range=(3, 7)):

        train_size = int(dataset_size * (1 -
                         (test_set_portion + valid_set_portion)))

        if train_size < 0:
            raise Exception("Illegal test and train set portions")

        test_size = int((dataset_size - train_size) * test_set_portion /
                        (test_set_portion + valid_set_portion))

        valid_size = dataset_size - train_size - test_size

        h = int(0.25*train_size)
        v = int(0.25*train_size)
        cr = int(0.25*train_size)
        ci = train_size - (h+v+cr)

        train = DataGenerator.GetHorizontalBars(h, image_size, noise, y_range, side_boundaries) + \
            DataGenerator.GetVerticalBars(
                v, image_size, noise, x_range, side_boundaries) + DataGenerator.GetCrosses(
                cr, image_size, noise, x_range, y_range, radius_range) + DataGenerator.GetCircles(
                ci, image_size, noise, x_range, y_range, radius_range)

        h = int(0.25*test_size)
        v = int(0.25*test_size)
        cr = int(0.25*test_size)
        ci = test_size - (h+v+cr)

        test = DataGenerator.GetHorizontalBars(h, image_size, noise, y_range, side_boundaries) + \
            DataGenerator.GetVerticalBars(
                v, image_size, noise, x_range, side_boundaries) + DataGenerator.GetCrosses(
                cr, image_size, noise, x_range, y_range, radius_range) + DataGenerator.GetCircles(
                ci, image_size, noise, x_range, y_range, radius_range)

        h = int(0.25*valid_size)
        v = int(0.25*valid_size)
        cr = int(0.25*valid_size)
        ci = valid_size - (h+v+cr)

        valid = DataGenerator.GetHorizontalBars(h, image_size, noise, y_range, side_boundaries) + \
            DataGenerator.GetVerticalBars(
                v, image_size, noise, x_range, side_boundaries) + DataGenerator.GetCrosses(
                cr, image_size, noise, x_range, y_range, radius_range) + DataGenerator.GetCircles(
                ci, image_size, noise, x_range, y_range, radius_range)

        np.random.shuffle(train)
        np.random.shuffle(test)
        np.random.shuffle(valid)
        return DataSet(train), DataSet(test), DataSet(valid)

    @staticmethod
    def show20Random(dataset):
        plt.figure(figsize=(20, 20))

        for i, j in enumerate(np.random.randint(0, len(dataset.data), 20)):
            plt.subplot(5, 5, i+1)
            plt.imshow(dataset.data[j].img)
