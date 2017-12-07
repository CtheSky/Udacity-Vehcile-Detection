import cv2
import numpy as np
from scipy.ndimage.measurements import label


def add_heat(heat_map, bbox_list):
    """Add heat to each bounding box and return the heat map image."""
    for box in bbox_list:
        heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heat_map


def apply_threshold(heat_map, threshold):
    """Suppress pixel value <= threshold and return the image."""
    heat_map[heat_map <= threshold] = 0
    return heat_map


def draw_labeled_bboxes(img, labels):
    """Find & draw labeled bounding boxes and return the image."""
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
        area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
        if area > 3200:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img


def get_heat_map(img, windows):
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, windows)
    heat_map = np.clip(heat, 0, 255)

    return heat_map


def get_labeled_image(img, heat_map):
    labels = label(heat_map)
    return draw_labeled_bboxes(np.copy(img), labels)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import io
    from sliding_window import *

    fig, ax = plt.subplots(3, 2)
    for i in range(0, 3):
        for j in range(0, 2):
            image = io.imread('test_images/test{}.jpg'.format(i * 2 + j + 1))

            windows = multi_scale_windows(image)
            windows = search_windows(image, windows, classifier.get_classifier())
            window_img = get_labeled_image(image, windows)

            ax[i, j].imshow(window_img)

    plt.show()
