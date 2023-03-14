# import packages
import cv2
import numpy as np;
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Tuple, List
import os
import re

file_dir = "/Users/huayinluo/Desktop/code/CaTracking" # change to your directory

def get_frame_coords(frame: int, neuron_allframes: List) -> List:
    """ Returns coords of neuron at given frame

    Parameters:
    frame: frame number
    neuron_allframes: array with 2 dimensions (frames, position)

    Returns:
    array with 2 dimensions (x, y)
    """
    return (np.ceil(neuron_allframes[frame,:])).astype(int)

def crop_image(imgs: List, frame: int, channel: int, position: Tuple[int, int], crop_size: int) -> List:
    """ Returns cropped grayscale image

    Parameters:
    frame: frame number
    position: (x,y) coord of neuron
    crop_size: (crop height, crop width)

    Returns:
    img: image in two dimension array (width, height)
    """
    (img_y, img_x) = imgs[frame, channel, :, :].shape # image dimensionss
    # avoid cropping out of bounds
    img = imgs[frame,
                channel,
                max(position[1]-crop_size, 0):min(position[1]+crop_size+1, img_y), # height (top:bottom)
                max(position[0]-crop_size, 0):min(position[0]+crop_size+1, img_x) # width (left:right)
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
    keypoints = detector.detect(thres_img)
    # (x,y) = keypoints[0].pt # when multiple keypoints, get coordinate, compare with position to see which keypoint is neuron

    # Draw keypoints
    im_with_keypoints = cv2.drawKeypoints(img, # input image
                                            keypoints, # keypoints
                                            img, # output image: set as same image to avoid error
                                            (0,0,255), # circle colour 
                                            3 # flag to draw with
                                            )
    
    return im_with_keypoints
            
# Load Data
imgs = np.load("11408_crop.nd2.npy")
ava = np.load("AVA_11408.mat.npy")
avb = np.load("AVB_11408.mat.npy")

start_frame = 0 #imgs.shape[0]
end_frame = 201
print(f"{end_frame-start_frame} frames")
channel = 0
(img_y, img_x) = imgs[0, 0, :, :].shape

crop_size = 10
thres = 50

crop = True # to skip cropping, set to False
blob = True # to skip blob detection, set to False
save = True # to save as npy file, set to True

# Crop Images
if crop:
    print("Start Cropping")
    try:
        for frame in range(start_frame, end_frame):
            # save original image (to check manually against final product)
            plt.imsave(f'{file_dir}/og_imgs/{frame}.png', imgs[frame,0,:,:], cmap="gray")

            # neuron positions
            position_a = get_frame_coords(frame, ava)
            position_b= get_frame_coords(frame, avb)
            
            # crop image & save
            img_a = crop_image(imgs, frame, channel, position_a, crop_size)
            img_b = crop_image(imgs, frame, channel, position_b, crop_size)
            
            plt.imsave(f'{file_dir}/cropped_imgs/{frame}a.png', img_a, cmap="gray")
            plt.imsave(f'{file_dir}/cropped_imgs/{frame}b.png', img_b, cmap="gray") 
                   
            print(f"------- cropped frame {frame}-------")
    except Exception as e:
        print(f"error cropping frame {frame}")
        print(e)
        pass
    print("End Cropping")


# Blob Detection
if blob:
    print("Start Blob Detection")
    npy_imgs=[]
    for frame in range(start_frame, end_frame):
        # Read images        
        img_gray_a = cv2.imread(f"{file_dir}/cropped_imgs/{frame}a.png")
        img_gray_b = cv2.imread(f"{file_dir}/cropped_imgs/{frame}b.png") 
        
        # Threshold image
        ret_a, thres_img_a = cv2.threshold(img_gray_a, thres, 255, cv2.THRESH_BINARY)
        ret_b, thres_img_b = cv2.threshold(img_gray_b, thres, 255, cv2.THRESH_BINARY)     
        
        # Save threshold images
        plt.imsave(f'{file_dir}/thres_imgs/{frame}a.png', thres_img_a) # save threshold image
        plt.imsave(f'{file_dir}/thres_imgs/{frame}b.png', thres_img_b)
        
        full_img = np.zeros((img_y, img_x, 3))
        (xa, ya) = get_frame_coords(frame, ava)
        (xb, yb) = get_frame_coords(frame, avb)
        # Paste into original image
        for i in range(thres_img_a.shape[0]):            
            row_a = full_img[min(max(ya-crop_size + i, 0), img_y)]
            row_b = full_img[min(max(yb-crop_size + i, 0), img_y)]
            
            row_a[max(xa-crop_size, 0):min(xa+crop_size+1, img_x)] = thres_img_a[i]
            row_b[max(xb-crop_size, 0):min(xb+crop_size+1, img_x)] = thres_img_b[i]
            
        norm_full_img = cv2.normalize(src=full_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.imsave(f'{file_dir}/full_imgs/{frame}.png', norm_full_img )
        npy_imgs.append(norm_full_img)
        print(f"-------finished frame {frame}-------")
    if save:
        imgset=np.array(npy_imgs)
        np.save("imgds.npy",imgset)       
    print("Finished!")
    