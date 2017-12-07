import numpy as np
from collections import deque
from moviepy.editor import VideoFileClip

import classifier
from sliding_window import search
from heat_map import get_heat_map, get_labeled_image, apply_threshold


heat_maps = deque(maxlen=5)
clf = classifier.get_classifier()


def process_video_frame(img):
    """Process the video frame and return the result image."""
    windows = search(img, clf)
    heat_map = get_heat_map(img, windows)
    heat_maps.appendleft(heat_map)

    merged_heat_map = np.zeros_like(heat_map)
    for heat_map in heat_maps:
        merged_heat_map += heat_map

    heat_map = apply_threshold(merged_heat_map, len(heat_maps) * 2)
    labeled_img = get_labeled_image(img, heat_map)

    return labeled_img


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Process video.')
    parser.add_argument('-f', default='project_video.mp4', help='path to video')
    parser.add_argument('-o', default=None, help='path to output video')
    args = parser.parse_args()

    clip2 = VideoFileClip(args.f)
    vid_clip = clip2.fl_image(process_video_frame)

    output = args.o if args.o else args.f.replace('.', '_output.')
    vid_clip.write_videofile(output, audio=False)
