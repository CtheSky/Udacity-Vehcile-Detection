import classifier
from sliding_window import search
from heat_map import get_heat_map, get_labeled_image, apply_threshold


def process_image(img, debug=False):
    clf = classifier.get_classifier()
    windows = search(img, clf)
    heat_map = get_heat_map(img, windows)
    heat_map = apply_threshold(heat_map, 1)
    labeled_img = get_labeled_image(img, heat_map)

    if debug:
        import matplotlib.pyplot as plt
        from sliding_window import draw_boxes

        f, axarr = plt.subplots(2, 2, figsize=(15, 15))

        axarr[0, 0].set_title('Original Image')
        axarr[0, 0].imshow(img)

        axarr[0, 1].set_title('Detected Windows')
        axarr[0, 1].imshow(draw_boxes(img, windows))

        axarr[1, 0].set_title('Heat Map')
        axarr[1, 0].imshow(heat_map)

        axarr[1, 1].set_title('Labeled Image')
        axarr[1, 1].imshow(labeled_img)

        plt.show()

    return labeled_img


if __name__ == '__main__':
    import argparse
    from skimage import io

    parser = argparse.ArgumentParser('Display processed image.')
    parser.add_argument('-f', default='test_images/test1.jpg', help='path to test image')
    parser.add_argument('-d', default=False, help='show result of each stage of pipeline')
    parser.add_argument('-o', default=None, help='path to output image')
    args = parser.parse_args()

    image = io.imread(args.f)
    result = process_image(image, debug=args.d)

    output = args.o if args.o else args.f.replace('.', '_output.')
    io.imsave(output, result)



