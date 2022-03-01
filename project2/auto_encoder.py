
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import time


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
        latent_size = 10

        # Define encoder
        encoder_input = keras.Input(shape=(28, 28, 1), name="image_input")
        x = layers.Conv2D(32, 5, strides=1, padding=padding,
                          activation="relu")(encoder_input)
        x = layers.Conv2D(64, 3, strides=2,
                          activation='relu', padding=padding)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, 3, strides=1,
                          activation='relu', padding=padding)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Flatten()(x)
        s = x.shape[-1]
        encoder_output = layers.Dense(latent_size, activation='relu')(x)

        encoder = keras.Model(encoder_input, encoder_output, name="encoder")

        # Define decoder
        decoder_input = keras.Input(shape=latent_size, name='decoder_input')

        x = layers.Dense(s, activation='relu')(decoder_input)
        x = layers.Reshape((14, 14, 64))(x)
        x = layers.Conv2DTranspose(
            32, 5, strides=1, padding='same', activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2DTranspose(
            32, 3, strides=1, padding='same', activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(
            16, 3, strides=2, padding='same', activation="relu")(x)
        decoder_output = layers.Conv2DTranspose(
            1, 3, strides=1, padding='same', activation='sigmoid')(x)

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

        self.latent_size = latent_size

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
            self.autoencoder.fit(x=x_train, y=x_train, shuffle=True, batch_size=512, epochs=epochs,
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

        return np.concatenate(channels, axis=1)

    def generate_images(self, n: int, no_channels=1):
        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError(
                "Model is not trained, so makes no sense to try to use it")
        channels = []
        for channel in range(no_channels):
            z = np.random.rand(n, self.latent_size) * 20
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
