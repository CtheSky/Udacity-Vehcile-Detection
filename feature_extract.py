import cv2
import os
import numpy as np
from skimage import feature

import load_dataset


feature_funcs = []


def use_feature(func):
    """Decorator to register feature funcs which accept an RGB image and return 1d array feature"""
    feature_funcs.append(func)
    return func


@use_feature
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


@use_feature
def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Return the RGB features of the given image"""
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))


@use_feature
def hog_feature(img, feature_vectore=True):
    """Return the HOG features of the given image"""

    # parameters to tune
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    features = [feature.hog(img[:, :, channel], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block), feature_vector=feature_vectore)
                for channel in range(3)]

    if feature_vectore:
        return np.concatenate(features)
    else:
        return features


def get_combined_feature(img):
    """Return the combined & registered features of the image."""
    return np.concatenate([func(img) for func in feature_funcs])


def get_features(use_cache=True):
    """Return extracted features of images and cache the result"""
    filename = 'features.npy'

    if use_cache and os.path.exists(filename):
        return np.load(filename)
    else:
        X = load_dataset.get_images()

        features = [get_combined_feature(x) for x in X]
        features = np.array(features)

        np.save(filename, features)
        return features


if __name__ == '__main__':
    get_features(use_cache=False)


