import cv2
import numpy as np
import feature_extract
import classifier


def search(img, classifier):
    """Return a list detected windows by running classifier on image."""
    windows = multi_scale_windows(img)
    return search_windows(img, windows, classifier)


def search_windows(img, size2windows, classifier):
    """
    Given an image and a list of bounding windows, run the
    classifier on the sub image and return windows predicted to be True
    """

    # Version 1: Without reusing hog feature extraction for windows with same size
    # good_windows = []
    # for window in windows:
    #     sub_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
    #     feature = feature_extract.get_combined_feature(sub_img)
    #
    #     if classifier.decision_function(feature.reshape(1, -1)) > 0.4:
    #         good_windows.append(window)
    # return good_windows

    # Version 2: reusing hog feature extraction for windows with same size
    #            hard code a little about hog size, improve it later
    good_windows = []

    base_size = 64
    for size, windows in size2windows.items():
        scale = size / base_size
        to_search = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))

        hog_ch1, hog_ch2, hog_ch3 = feature_extract.hog_feature(to_search, feature_vectore=False)
        for window in windows:
            x_start = int(window[0][0]/scale)
            y_start = int(window[0][1]/scale)
            x_end = x_start + base_size
            y_end = y_start + base_size
            x_hog_start = int(x_start/8)
            y_hog_start = int(y_start/8)

            # compute feature for window
            feature = []
            feature.append(feature_extract.bin_spatial(to_search[y_start:y_end, x_start:x_end, :]))
            feature.append(feature_extract.color_hist(to_search[y_start:y_end, x_start:x_end, :]))
            feature.append(hog_ch1[y_hog_start:y_hog_start + 7, x_hog_start: x_hog_start + 7, ...].ravel())
            feature.append(hog_ch2[y_hog_start:y_hog_start + 7, x_hog_start: x_hog_start + 7, ...].ravel())
            feature.append(hog_ch3[y_hog_start:y_hog_start + 7, x_hog_start: x_hog_start + 7, ...].ravel())
            feature = np.concatenate(feature)

            if classifier.decision_function(feature.reshape(1, -1)) > 0.4:
                good_windows.append(window)

    return good_windows


def multi_scale_windows(img):
    """Return dict of size -> a list of window positions"""
    size2windows = {
        64:  slide_window(img, y_top=380, y_bottom=500, xy_window=(64, 64), xy_overlap=(0.7, 0.8)),
        85:  slide_window(img, y_top=380, y_bottom=500, xy_window=(85, 85), xy_overlap=(0.7, 0.8)),
        100: slide_window(img, y_top=400, y_bottom=600, xy_window=(100, 100), xy_overlap=(0.7, 0.8)),
        200: slide_window(img, y_top=400, y_bottom=650, xy_window=(200, 200), xy_overlap=(0.7, 0.8))
    }
    return size2windows


def slide_window(img, x_left=None, x_right=None, y_top=None, y_bottom=None,
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Generate and return a list of windows given the window parameters."""
    # If x and/or y start/stop positions not defined, set to image size
    if not x_left:
        x_left = 0
    if not x_right:
        x_right = img.shape[1]
    if not y_top:
        y_top = 0
    if not y_bottom:
        y_top = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_right - x_left
    yspan = y_bottom - y_top

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            x_start = xs * nx_pix_per_step + x_left
            x_end = x_start + xy_window[0]
            y_start = ys * ny_pix_per_step + y_top
            y_end = y_start + xy_window[1]

            window_list.append(((x_start, y_start), (x_end, y_end)))

    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw bounding boxes and return the image."""
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import io

    # image = io.imread('test_images/test1.jpg')
    # feature_extract.get_combined_feature(image[0:64, 0:64, :])

    fig, ax = plt.subplots(3, 2)
    for i in range(0, 3):
        for j in range(0, 2):
            image = io.imread('test_images/test{}.jpg'.format(i * 2 + j + 1))

            windows = multi_scale_windows(image)
            windows = search_windows(image, windows, classifier.get_classifier())
            window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)

            ax[i, j].imshow(window_img)

    plt.show()