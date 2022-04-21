import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RNN:

    def __init__(self, k, no_features) -> None:
        self.k = k
        input_layer = keras.Input(shape=(k, no_features), name='Input layer')

        x = layers.LSTM(50, return_sequences=True)(input_layer)
        x = layers.Dropout(0.1)(x)
        x = layers.LSTM(50)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(100, activation='relu')(x)

        output_layer = layers.Dense(1, activation='linear')(x)

        model = keras.Model(input_layer, output_layer, name='LSTM_Model')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=keras.losses.MeanSquaredError()
        )
        self.model = model

    def train(self, x_train, y_train, x_valid, y_valid, epochs=10, batch_size=128) -> None:

        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, y_valid))

    def forecast(self, x, n=20, s=0):

        preds = np.zeros(s+n)
        x_t = x.copy()
        # model(x_test[:w])
        for i in range(n):
            p = self.model(x_t[s+i:s+i+1])[0][0]

            for j in range(i, i+self.k):
                # Replace "previous_y" for the next k rows
                x_t[s+j+1][-(1+j-i)][-1] = p
            preds[s+i] = p
        return preds

    def predict(self, x):

        return self.model.predict(x)

    def plot_train_history(self):
        pass

    def forecast_n(self, x, n, plot=True):
        pass
