import matplotlib.pyplot as plt
import numpy as np


def plot_10_images(original, output, filename=None, random=False):
    plt.figure(figsize=(20, 4))

    n = 10
    for i in range(1, n + 1):
        ind = i

        if random:
            ind = np.random.randint(0, len(original))
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(original[ind])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(output[ind])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if filename:
        plt.savefig('plots/'+filename)
    plt.show()
