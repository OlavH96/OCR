import os
import numpy as np
import imageio
import glob


def load() -> [np.ndarray, [str]]:
    print("Loading Data...")

    data_dir = os.path.join('.', '..', 'data', 'chars74k-lite')

    data = []
    labels = []

    for directory in sorted(list(filter(lambda x: len(x) == 1, os.listdir(data_dir)))):
        dir_path = os.path.join(data_dir, directory)
        for file in glob.glob(dir_path + "/*.jpg"):
            im = imageio.imread(file)
            data.append(im)
            labels.append(directory)

    data = np.array(data)
    print("Finished Loading Data")

    return data, labels


def load_detection_images(path) -> np.ndarray:
    data = []

    for file in glob.glob(path + "/*.jpg"):
        im = imageio.imread(file)
        data.append(im)

    data = np.array(data)

    return data
