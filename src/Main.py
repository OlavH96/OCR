from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

import src.model.Loader as Loader
import src.model.Prepare as Prepare
import src.model.Labels as Labels
import src.model.Train as Train
import matplotlib.pyplot as plt
import numpy as np


def train_nn(epochs=10):
    data, labels = Loader.load()
    labels = np.array([Labels.from_char(l) for l in labels])

    data, labels = Prepare.shuffle(data, labels)
    data: np.ndarray = Prepare.normalize(data)
    train_data, test_data, train_labels, test_labels = Prepare.split(data, labels, percent_test=0.2)
    train_data = Prepare.noise_removal(train_data)
    train_data = Prepare.expand_dims(train_data)
    test_data = Prepare.expand_dims(test_data)

    model = Train.build_model(input_shape=(data.shape[1], data.shape[2], 1), output_len=Labels.from_char('z') + 1)

    model = Train.train(train_data, train_labels, model, epochs=epochs)

    loss, accuracy = Train.evaluate(model, test_data, test_labels)
    print("Model Loss", loss)
    print("Model Accuracy", accuracy)

    if save_model:
        model.save('model.h5')

    if display_graphs:
        predict("NN", test_data, test_labels, model, two_d=True)


def train_knn():
    data, labels = Loader.load()
    labels = np.array([Labels.from_char(l) for l in labels])

    data, labels = Prepare.shuffle(data, labels)
    data: np.ndarray = Prepare.normalize(data)
    # data = Prepare.noise_removal(data)
    data = Prepare.flatten(data)
    train_data, test_data, train_labels, test_labels = Prepare.split(data, labels, percent_test=0.2)
    train_data = Prepare.noise_removal(train_data)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_data, train_labels)
    # get the model accuracy
    model_score = knn.score(test_data, test_labels)
    print("KNN Score", model_score)

    if save_model:
        joblib.dump(knn, 'knn_model.pkl')

    if display_graphs:
        predict("KNN", test_data, test_labels, knn, two_d=False)


def predict(prefix, data, labels, model, two_d=True):
    p_data = data[:10]
    labels = labels[:10]
    predict = model.predict(p_data)
    if two_d:
        predict = predict.argmax(axis=1)

    for x, p, l in zip(p_data, predict, labels):
        x: np.ndarray = x

        x = np.reshape(x, newshape=(20, 20))

        plt.imshow(x, cmap='gray')
        plt.title(prefix+": actual="+Labels.from_int(l)+", predicted="+Labels.from_int(p))
        plt.show()


if __name__ == '__main__':
    save_model = True
    display_graphs = True

    train_knn()
    train_nn(epochs=20)
