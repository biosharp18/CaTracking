import cv2
import os
import natsort
import ffmpeg

image_folder = "/Users/huayinluo/Desktop/code/CaTracking/masked/11413" # change to your directory
(
    ffmpeg
    .input(f'{image_folder}/*.png', pattern_type='glob', framerate=25)
    .output('movie.mp4')
    .run()
)