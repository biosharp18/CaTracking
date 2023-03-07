# import packages
import cv2
import numpy as np;
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from typing import Tuple, List
import re

file_dir = "/Users/huayinluo/Desktop/code/CaTracking/blob_detection" # change to your directory


def get_frame_coords(frame: int, neuron_allframes: List) -> List:
    """ Returns coords of neuron at given frame

    Parameters:
    frame: frame number
    neuron_allframes: array with 2 dimensions (frames, position)

    Returns:
    array with 2 dimensions (x, y)
    """
    return (np.ceil(neuron_allframes[frame,:])).astype(int)

def crop_image(imgs: List, frame: int, position: Tuple[int, int], crop_size: Tuple[int, int]) -> List:
    """ Returns cropped grayscale image

    Parameters:
    imgs: the video (list of frames)
    frame: frame number
    position: (x,y) coord of neuron
    crop_size: (crop height, crop width)

    Returns:
    img: image in two dimension array (width, height)
    """
    channel = 0
    dimensions = imgs[frame, channel, :, :].shape # image dimensionss
    img = imgs[frame,
                channel,
                max(position[0]-crop_size[0], 0):min(position[0]+crop_size[0], dimensions[0] - 1), # height (top:bottom)
                max(position[1]-crop_size[1], 0):min(position[1]+crop_size[1], dimensions[1] - 1) # width (left:right)
                ]
    return img

def draw_circle(img: List, thres_img: List, position: List) -> List:
    """ Draw circle around neuron

    Parameters:
    img: image to draw circle on
    thres_img: thresholded image to find keypoints on
    position: coordinates of neuron

    Returns:
    im_with_keypoints: array of image with circled neuron
    """
    # Set params for detecter
    params = cv2.SimpleBlobDetector_Params() 
    # params.minThreshold = 25 # lower than this, turn into black (0)
    # params.maxThreshold = 300 # higher than this, turn into black (0)
    # params.filterByCircularity = True
    # params.minCircularity = 0.1

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect keypoints
    keypoints = detector.detect(thres_img)\
    # (x,y) = keypoints[0].pt # when multiple keypoints, get coordinate, compare with position to see which keypoint is neuron

    # Draw keypoints
    im_with_keypoints = cv2.drawKeypoints(img, # input image
                                            keypoints, # keypoints
                                            img, # output image: set as same image to avoid error
                                            (0,0,255), # circle colour 
                                            3 # flag to draw with
                                            )
    
    return im_with_keypoints

def contour_detection(img, thres_img, min_area):
    contours = cv2.findContours(thres_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    line_length = 20
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00']) 
            cv2.line(img, (x-line_length, y), (x+line_length, y), (255, 0, 0), 2)
            cv2.line(img, (x, y-line_length), (x, y+line_length), (255, 0, 0), 2)
    return img
            
# Load Data
imgs = np.load("11408_crop.nd2.npy")
ava = np.load("AVA_11408.mat.npy")
avb = np.load("AVB_11408.mat.npy")

frames = 5 # frames = imgs.shape[0] to run for all frames
crop_size = (30,30)
chosen_av = avb
crop = False # to skip cropping, set to False
blob = True # to skip blob detection, set to False

# Crop Images
if crop:
    print("Start Cropping")
    try:
        for frame in range(frames):
            position = get_frame_coords(frame, chosen_av)
            img = crop_image(imgs, frame, position, crop_size) # crop image
            plt.imsave(f'{file_dir}/cropped_imgs/cropped_{frame}.png', img, cmap="gray") # save cropped image
            
            print(f"position: {position}")
            print(f"------- cropped frame {frame}-------")
    except:
        print(f"error cropping frame {frame}")
        pass
    print("End Cropping")


# Blob Detection
if blob:
    print("Start Blob Detection")
    for frame in range(frames):
        try:         
            img_gray = cv2.imread(f"{file_dir}/cropped_imgs/cropped_{frame}.png") # read image
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
            
            ret, thres_img = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV) # threshold & invert image
            blob_img = draw_circle(img_gray, thres_img, get_frame_coords(frame, chosen_av)) # use blob detection
            
            # blob_img = contour_detection(img, img_gray, 20) # use contour detection (doesn't work yet, incompatible array)
            
            plt.imsave(f'{file_dir}/blob_imgs/blob_img{frame}.png', blob_img) # save image
            
            print(f"-------finished frame {frame}-------")
        except:
            print(f"error blob detection frame {frame}")
            pass           
    print("Finished!")