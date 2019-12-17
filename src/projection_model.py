import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from networkx import shortest_path_length
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
#%%


class ProjectionModel:
    def __init__(self):
        self.model = self._construct_model()

    def _construct_model(self):
        model = Sequential()
        model.add(Dense(100, activation=None))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
        return model

    def train(self, X_train, Y_train, epochs=300):
        history = self.model.fit(X_train, Y_train,
                                 epochs=epochs, verbose=0).history

        plt.plot(history['loss'])
        plt.show()
        return history

    def predict(self, X_test, node_embeddings):
        pred_vectors = normalize(self.model.predict(X_test))
        return [node_embeddings.similar_by_vector(vector, topn=1)[0][0] for vector in pred_vectors]

    def evaluate(self, truths, preds, graph):
        accuracy = accuracy_score(truths, preds)
        print(f'Accuracy: {accuracy}')

        sps = [shortest_path_length(graph, source=pred, target=truth) for truth, pred in zip(truths, preds)]

        sns.distplot(sps, bins=np.arange(0, max(sps) + 1, 1))
        plt.xticks(range(-2, max(sps) + 1, 1))
        plt.title('Shortest Path Length Distribution')
        plt.show()

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, load_path):
        self.model = load_model(load_path)
