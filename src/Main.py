from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

import src.model.Loader as Loader
import src.model.Prepare as Prepare
import src.model.Labels as Labels
import src.model.Train as Train
import matplotlib.pyplot as plt
import numpy as np


def train_nn():
    model = Train.build_model(input_shape=(data.shape[1], data.shape[2], 1), output_len=Labels.from_char('z') + 1)

    model = Train.train(train_data, train_labels, model, epochs=25)

    loss, accuracy = Train.evaluate(model, test_data, test_labels)
    print("Model Loss", loss)
    print("Model Accuracy", accuracy)

    model.save('model.h5')


def train_knn():
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_data, train_labels)
    # get the model accuracy
    model_score = knn.score(test_data, test_labels)
    print(model_score)
    joblib.dump(knn, 'knn_model.pkl')

    predict = knn.predict(test_data[:10])
    print(predict)
    for x, y,p in zip(test_data[:10], labels[:10], predict):
        x: np.ndarray = x
        x = np.reshape(x, newshape=(20,20))

        plt.imshow(x, cmap='gray')
        plt.title(Labels.from_int(p))
        plt.show()


if __name__ == '__main__':
    data, labels = Loader.load()
    labels = np.array([Labels.from_char(l) for l in labels])

    data, labels = Prepare.shuffle(data, labels)
    data: np.ndarray = Prepare.normalize(data)
    data = Prepare.noise_removal(data)
    data = Prepare.flatten(data)
    train_data, test_data, train_labels, test_labels = Prepare.split(data, labels, percent_test=0.2)
    # train_data = Prepare.expand_dims(train_data)
    # test_data = Prepare.expand_dims(test_data)

    # train_nn()
    train_knn()
