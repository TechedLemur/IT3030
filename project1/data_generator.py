import numpy as np


class DataGenerator:

    @staticmethod
    def HorizontalBar(size: int = 16):

        img = np.zeros((size, size), dtype=np.bool)

        for i in np.random.choice(np.arange(size), np.random.randint(1, int(size/3))):
            img[i, :] = np.True_
        return img

    @staticmethod
    def VerticalBar(size: int = 16):

        img = np.zeros((size, size))

        for i in np.random.choice(np.arange(size), np.random.randint(1, int(size/3))):
            img[:, i] = 1
        return img

    # Returns an image with added random noise with probability p

    @staticmethod
    def AddNoise(img, p=0.05):

        mask = np.random.choice(a=[True, False], size=img.shape, p=[p, 1-p])

        return np.bitwise_xor(img, mask)
