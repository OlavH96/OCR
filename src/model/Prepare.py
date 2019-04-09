import numpy as np


def normalize(data: np.ndarray) -> np.ndarray:
    normalized = data / data.max()

    return normalized


def flatten(data: np.ndarray) -> np.ndarray:
    s = data.shape
    flattened = data.reshape((s[0], s[1] * s[2]))

    return flattened
