import operator
import os
import functools as funcs
import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_windows

from src.model import Loader, Prepare, Labels
import math

DETECTION_IMAGES_DIR = os.path.join('..', 'data', 'detection-images')
CUTOFF = 0.2


def euclidian(x1, y1, x2, y2) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# This filters out windows with only parts of the letter showing
# Since the model might be very confident that half of a 'A' is a 'J'
# So using confidence cutoff only does not work very well
def white_pixel_filter(data: np.ndarray) -> bool:
    temp = data - 1
    temp = np.abs(temp)
    num_pixels = funcs.reduce(operator.mul, temp.shape)
    num_none_white_pixels = np.count_nonzero(temp)
    limit = num_pixels / 1.5
    return num_none_white_pixels < int(limit)


# Removed predictions with very close other predictions, since in reality
# you never want overlapping letters
# Selects the closest letter with the highest certainty using euclidean distance
def contention_filter(indexes, range_of_contention):
    result = []

    for index in range(len(indexes)):
        i, j, label, certainty = indexes[index]
        contention = []
        for index1 in range(len(indexes)):
            if index1 == index: continue

            i1, j1, label1, certainty1 = indexes[index1]
            distance = euclidian(i, j, i1, j1)
            if distance < range_of_contention:
                contention.append((i1, j1, label1, certainty1))
                contention.append((i, j, label, certainty))

        if len(contention) == 0:
            result.append((i, j, label, certainty))
        else:
            best = max(contention, key=lambda x: x[3])  # break contention on certainty
            if best not in result:
                result.append(best)

    return result


def sliding_window_prediction(d, step):
    windows = view_as_windows(d, window_shape=(window_x, window_y), step=step)
    indexes = []
    for i, p in enumerate(windows):
        for j, w in enumerate(p):

            should_skip = white_pixel_filter(w)
            if should_skip: continue

            w = np.reshape(w, newshape=(1, window_x, window_y, 1))
            prediction = model.predict(w)
            certainty = prediction.max()

            if certainty < CUTOFF: continue

            predict = prediction.argmax(axis=1)
            label = Labels.from_int(predict.tolist()[0])

            indexes.append((i, j, label, certainty))
    return indexes


def draw(d, result, word):
    fig, ax = plt.subplots(1)
    ax.imshow(d, cmap='gray', vmin=0, vmax=1)
    for i, j, label, certainty in result:
        print(i, j, label, certainty)

        x = j * window_step
        y = i * window_step

        rect = patches.Rectangle((x, y), window_x, window_y, linewidth=1, edgecolor='r', facecolor='none')
        ax.text(x, y - 10, label)
        ax.add_patch(rect)
    plt.title(word)
    plt.show()


def construct_word(data, first_letter_highest=False):
    if first_letter_highest:
        data = list(sorted(data, key=lambda x: (x[0], x[1])))
    else:
        data = list(sorted(data, key=lambda x: (x[1], x[0])))
    res = list(map(lambda x: x[2], data))

    return "".join(res)


if __name__ == '__main__':
    model: keras.models.Sequential = keras.models.load_model('model.h5')
    window_shape = model.input_shape
    window_x = window_shape[1]
    window_y = window_shape[2]

    window_step = 6

    print("Window Shape", window_shape)

    data: [np.ndarray] = Loader.load_detection_images(DETECTION_IMAGES_DIR)

    data = np.array([Prepare.normalize(d) for d in data])

    small = data[0]
    large = data[1]

    indexes = sliding_window_prediction(small, window_step)
    result = contention_filter(indexes, range_of_contention=int(window_step))
    word = construct_word(result)
    draw(small, result, word)

    indexes = sliding_window_prediction(large, window_step)
    result = contention_filter(indexes, range_of_contention=int(window_step - 2))
    word = construct_word(result, first_letter_highest=True)
    draw(large, result, word)
