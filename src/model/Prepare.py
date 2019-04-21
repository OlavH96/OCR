import numpy as np
from sklearn.utils import shuffle as skshuffle
from src.OCR import white_pixel_filter
import matplotlib.pyplot as plt


def normalize(data: np.ndarray) -> np.ndarray:
    normalized = data / data.max()

    return normalized


def flatten(data: np.ndarray) -> np.ndarray:
    s = data.shape
    flattened = data.reshape((s[0], s[1] * s[2]))

    return flattened


def split(data: np.ndarray, labels, percent_test) -> (np.ndarray, np.ndarray, [int], [int]):
    l = data.shape[0]
    i_test = int(l * percent_test)
    test, train = np.split(data, [i_test])

    train_labels = labels[i_test:]
    test_labels = labels[:i_test]
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train, test, train_labels, test_labels


def expand_dims(data: np.ndarray) -> np.ndarray:
    return np.expand_dims(data, len(data.shape))


def shuffle(x: np.ndarray, y: np.ndarray):
    x, y = skshuffle(x, y)
    return x, y


def noise_removal(data: np.ndarray):
    data[data > 0.5] = 1
    data[data < 0.5] = 0
    return data
