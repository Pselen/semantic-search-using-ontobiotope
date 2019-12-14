import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#%%


class ProjectionModel:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.model = self._construct_model()

    def _construct_model(self):
        model = Sequential()
        model.add(Dense(100, activation=None))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
        return model

    def train(self, epochs=300):
        history = model.fit(self.X_train, self.Y_train,
                            epochs=epochs, verbose=0).history

        plt.plot(history['loss'])
        plt.show()

    def save(self, save_path):
        self.model.save(save_path)
