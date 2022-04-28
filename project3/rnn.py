from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers


class RNN:

    def __init__(self, k, no_features, altered=False) -> None:
        self.k = k
        input_layer = keras.Input(shape=(k, no_features))

        lr = 0.000125

        x = layers.LSTM(30, return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(input_layer)
        x = layers.Dropout(0.13)(x)
        x = layers.LSTM(20,  return_sequences=altered, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(x)

        if altered:
            lr = 0.00013
            x = layers.Dropout(0.12)(x)
            x = layers.LSTM(25, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.L2(1e-4),
                            activity_regularizer=regularizers.L2(1e-5))(x)
        x = layers.Dropout(0.12)(x)
        x = layers.Dense(100, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.L2(1e-4),
                         activity_regularizer=regularizers.L2(1e-5))(x)

        output_layer = layers.Dense(1, activation='linear')(x)

        model = keras.Model(input_layer, output_layer, name='LSTM_Model')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            # loss=keras.losses.MeanSquaredError()
            loss=RNN.root_mean_squared_error


        )
        self.model = model

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def train(self, x_train, y_train, x_valid, y_valid, epochs=10, batch_size=128) -> None:

        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, y_valid))

    def forecast(self, x, n=20, s=0, replace=True):

        preds = np.zeros(s+n)
        x_t = x.copy()
        for i in range(n):
            p = self.model(x_t[s+i:s+i+1])[0][0]
            if replace:
                for j in range(i, min(i+self.k, i+n)):
                    # Replace "previous_y" for the next rows
                    x_t[s+j+1][-(1+j-i)][-1] = p
            preds[s+i] = p
        return preds

    def predict(self, x):

        return self.model.predict(x)

    def plot_train_history(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper right')
        plt.savefig("./figures/training_loss.png")
        plt.show()

    def forecast_15(self, x, y, M=100, n=20, replace=True, path="", benchmark=False):
        """
        Forecast and plot 15 random sequences of length n
        """
        plt.figure(figsize=(20, 20))
        for i in range(15):
            if benchmark:
                s = M + i*n
            else:
                s = np.random.randint(M, len(x)-M)
            p = self.forecast(x, n=n, s=s, replace=replace)

            t = np.arange(len(x))

            plt.subplot(3, 5, i+1)

            plt.plot(t[s-M:s+1], y[s-M:s+1])
            plt.plot(t[s:s+n], p[s:s+n])
            plt.plot(t[s:s+n], y[s:s+n])
            plt.legend(['y_historical', 'y_pred', 'y_target'],
                       loc='upper left')
        plt.savefig(path)
        plt.show()

    def forecast_1(self, x, y, M=100, n=20, s=None, replace=True, path=""):
        if not s:
            s = np.random.randint(M, len(x)-M)
        p = self.forecast(x, n=n, s=s, replace=replace)

        t = np.arange(len(x))

        plt.plot(t[s-M:s+1], y[s-M:s+1])
        plt.plot(t[s:s+n], p[s:s+n])
        plt.plot(t[s:s+n], y[s:s+n])
        plt.legend(['y_historical', 'y_pred', 'y_target'],
                   loc='upper left')
        plt.savefig(path)
        plt.show()

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path, new_k=None):
        self.model = keras.models.load_model(
            path, custom_objects={'root_mean_squared_error': RNN.root_mean_squared_error})
        if new_k:
            self.k = new_k
