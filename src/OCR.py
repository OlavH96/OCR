import operator
import os
from functools import reduce

import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_windows

from src.model import Loader, Prepare, Labels

DETECTION_IMAGES_DIR = os.path.join('..', 'data', 'detection-images')
CUTOFF = 0.7

if __name__ == '__main__':
    model: keras.models.Sequential = keras.models.load_model('model.h5')
    window_shape = model.input_shape
    window_x = window_shape[1]
    window_y = window_shape[2]

    window_step = 6
    print("Window Shape", window_shape)

    data: [np.ndarray] = Loader.load_detection_images(DETECTION_IMAGES_DIR)

    data = np.array([Prepare.normalize(d) for d in data])

    for d in data[1:]:
        d: np.ndarray = d
        image_shape = d.shape

        # print("Image Shape", image_shape)
        # print("Image Pixels", image_pixels)
        # print("Num Windows", num_windows)

        windows = view_as_windows(d, window_shape=(window_x, window_y), step=window_step)
        print(windows.shape)
        indexes = []
        for i, p in enumerate(windows):
            for j, w in enumerate(p):
                temp = w-1
                temp = np.abs(temp)
                num_pixels = reduce(operator.mul, temp.shape)
                num_none_white_pixels = np.count_nonzero(temp)
                limit = num_pixels/1.5
                if num_none_white_pixels < int(limit):
                    # print("Skipping", num_pixels, num_none_white_pixels, limit)
                    continue

                w = np.reshape(w, newshape=(1, window_x, window_y, 1))
                prediction = model.predict(w)
                certainty = prediction.max()

                if certainty < CUTOFF:
                    # print("Cutoff", certainty)
                    continue

                predict = prediction.argmax(axis=1)
                label = Labels.from_int(predict.tolist()[0])
                print(label, certainty)
                indexes.append((i, j, label, certainty))

        fig, ax = plt.subplots(1)
        ax.imshow(d, cmap='gray', vmin=0, vmax=1)
        for i, j, label, certainty in indexes:
            x = j * window_step
            y = i * window_step

            rect = patches.Rectangle((x, y), window_x, window_y, linewidth=1, edgecolor='r', facecolor='none')
            ax.text(x, y-5, label)
            ax.add_patch(rect)
        plt.show()
