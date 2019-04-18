import os
import tensorflow as tf
import keras
from src.model import Loader, Prepare, Labels, Train
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import reduce
import operator
from scipy.ndimage.filters import maximum_filter1d
from skimage.util import view_as_windows

DETECTION_IMAGES_DIR = os.path.join('..', 'data', 'detection-images')
CUTOFF = 0.99

if __name__ == '__main__':
    model: keras.models.Sequential = keras.models.load_model('model.h5')
    window_shape = model.input_shape
    window_pixels = reduce(operator.mul, window_shape[1:])
    # print("Window Shape", window_shape)
    # print("Window Pixels", window_pixels)

    data: [np.ndarray] = Loader.load_detection_images(DETECTION_IMAGES_DIR)

    data = np.array([Prepare.normalize(d) for d in data])

    for d in data[:1]:
        d: np.ndarray = d
        image_shape = d.shape
        image_pixels = reduce(operator.mul, image_shape)
        num_windows = int(image_pixels / window_pixels)

        # print("Image Shape", image_shape)
        # print("Image Pixels", image_pixels)
        # print("Num Windows", num_windows)

        # windows = np.reshape(d, newshape=(num_windows, window_shape[1], window_shape[2]))
        # windows = blockshaped(d, 20, 20)
        # windows = np.lib.stride_tricks.as_strided(d, shape=(num_windows, 20, 20))
        # windows = np.reshape(windows, newshape=(num_windows, 20, 20))
        windows = view_as_windows(d, window_shape=(20, 20), step=5)
        print(windows.shape)
        indexes = []
        for i, p in enumerate(windows):
            for j, w in enumerate(p):

                w = np.reshape(w, newshape=(1, w.shape[0], w.shape[1], 1))
                prediction = model.predict(w)
                certainty = prediction.max()

                if certainty < CUTOFF:
                    continue

                predict = prediction.argmax(axis=1)
                label = Labels.from_int(predict.tolist()[0])

                indexes.append((i, j))

        fig, ax = plt.subplots(1)
        ax.imshow(d, cmap='gray', vmin=0, vmax=1)
        for i, j in indexes:
            rect = patches.Rectangle((j * 5, i * 5), 20, 20, linewidth=1, edgecolor='r', facecolor='none')

            ax.add_patch(rect)
        plt.show()
