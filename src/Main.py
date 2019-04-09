import src.model.Loader as Loader
import src.model.Prepare as Prepare
import src.model.Labels as Labels

import numpy as np

if __name__ == '__main__':
    data, labels = Loader.load()
    labels = [Labels.from_char(l) for l in labels]

    normalized_data: np.ndarray = Prepare.normalize(data)

    print(normalized_data.shape)
    print(Prepare.flatten(normalized_data).shape)

