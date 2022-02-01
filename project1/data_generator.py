import numpy as np


class DataGenerator:

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
