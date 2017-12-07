import os
import glob
import numpy as np
from skimage import io

vehicle_path = './data/vehicles/*/*'
non_vehicle_path = './data/non-vehicles/*/*'


def get_images():
    """Read data from given path and return two numpy array X, y"""
    filename = 'images.npy'

    if os.path.exists(filename):
        return np.load(filename)
    else:
        X = []

        for path in glob.glob(vehicle_path):
            X.append(io.imread(path))
        for path in glob.glob(non_vehicle_path):
            X.append(io.imread(path))

        X = np.array(X)
        np.save(filename, X)

        return X


def get_labels():
    """Read data from given path and return labels as a numpy array y"""
    filename = 'labels.npy'

    if os.path.exists(filename):
        return np.load(filename)

    else:
        y = []

        for _ in glob.glob(vehicle_path):
            y.append(1)
        for _ in glob.glob(non_vehicle_path):
            y.append(0)

        y = np.array(y).reshape(-1, 1)
        np.save(filename, y)

        return y
