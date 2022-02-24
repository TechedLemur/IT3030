
import numpy as np
from stacked_mnist import StackedMNISTData
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import time
import matplotlib.pyplot as plt
import pickle


class AutoEncoder():

    def __init__(self, force_learn: bool = False, file_name: str = "./models/ae_model") -> None:
        """
        Define model and set some parameters.
        The model is  made for classifying one channel only -- if we are looking at a
        more-channel image we will simply do the thing one-channel-at-the-time.
        """
        self.force_relearn = force_learn
        self.file_name = file_name

        padding = 'same'

        encoder_input = keras.Input(shape=(28, 28, 1), name="image_input")
        x = layers.Conv2D(16, 3, strides=(2, 2), padding=padding,
                          activation="relu")(encoder_input)

        encoder_output = layers.Conv2D(
            4, 3, strides=(2, 2), padding=padding, activation='sigmoid')(x)

        encoder = keras.Model(encoder_input, encoder_output, name="encoder")

        decoder_input = encoder_output
        x = layers.Conv2DTranspose(8, 2, strides=(
            2, 2), padding=padding, activation="relu")(decoder_input)
        x = layers.Conv2DTranspose(16, 3, strides=(
            2, 2), padding=padding, activation="relu")(x)
        decoder_output = layers.Conv2DTranspose(
            1, 3, padding=padding, activation='sigmoid')(x)

        decoder = keras.Model(decoder_input, decoder_output, name="decoder")

        autoencoder_input = keras.Input(shape=(28, 28, 1), name="image_input")
        encoded_img = encoder(autoencoder_input)
        decoded_img = decoder(encoded_img)
        autoencoder = keras.Model(
            autoencoder_input, decoded_img, name="autoencoder")

        autoencoder.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder

        self.done_training = self.load_weights()

    def load_weights(self):
        # noinspection PyBroadException
        if (self.force_relearn):
            print("Not trained")
            return False
        try:
            self.autoencoder.load_weights(filepath=self.file_name)

            print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(
                f"Could not read weights for autoencoder from file. Must retrain...")
            done_training = False

        return done_training

    def train(self, x_train, x_test, epochs: np.int = 10) -> bool:

        tensorboard_callback = TensorBoard(
            log_dir=f"./logs/{int(time.time())}")

        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
        self.done_training = self.load_weights()

        if not self.done_training:
            # Get hold of data

            # "Translate": Only look at "red" channel; only use the last digit. Use one-hot for labels during training
            self.no_channels = x_train.shape[-1]

            x_train = x_train[:, :, :, [0]]

            x_test = x_test[:, :, :, [0]]

            # Fit model
            self.autoencoder.fit(x=x_train, y=x_train, shuffle=True, batch_size=1024, epochs=epochs,
                                 validation_data=(x_test, x_test),  callbacks=[tensorboard_callback])

            # Save weights and leave
            self.autoencoder.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training

    def decode_channel(self, data: np.ndarray):

        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError(
                "Model is not trained, so makes no sense to try to use it")

        return self.decoder.predict(data)

    def encode(self, data: np.ndarray):
        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError(
                "Model is not trained, so makes no sense to try to use it")

        no_channels = data.shape[-1]
        channels = []
        for channel in range(no_channels):
            channels.append(self.encoder.predict(data[:, :, :, [channel]]))

        return np.concatenate(channels, axis=3)

    def generate_images(self, n: int):
        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError(
                "Model is not trained, so makes no sense to try to use it")
        channels = []
        for channel in range(self.no_channels):
            z = np.random.randn(n, 7, 7, 4)+0.5

            channels.append(self.decoder.predict(z))
        return np.concatenate(channels, axis=3)

    def autoencode(self, data: np.ndarray):

        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError(
                "Model is not trained, so makes no sense to try to use it")

        no_channels = data.shape[-1]
        channels = []
        for channel in range(no_channels):
            channels.append(self.autoencoder.predict(data[:, :, :, [channel]]))

        return np.concatenate(channels, axis=3)

    @staticmethod
    def plot_n_images(n, original, output):
        plt.figure(figsize=(20, 4))
        for i in range(1, n + 1):
            # Display original
            ax = plt.subplot(2, n, i)
            plt.imshow(original[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.imshow(output[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
