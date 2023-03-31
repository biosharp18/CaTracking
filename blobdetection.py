# import packages
import cv2
import numpy as np;
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Tuple, List
import time
# import os
# import re2

file_dir = "/Users/huayinluo/Desktop/code/CaTracking" # change to your directory
video_num = "11433" # change to video num 

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
    """ Returns cropped grayscale image of given frame and video

    Parameters:
    imgs: 2 channel video
    frame: frame number
    channel: channel number
    position: (x,y) coord of neuron
    crop_size: crop size

    Returns:
    img: image in two dimension array (width, height)
    """
    (img_y, img_x) = imgs[frame, channel, :, :].shape # image dimensionss
    # avoid cropping out of bounds
    gray_img = imgs[frame,
                channel,
                max(position[1]-crop_size, 0):min(position[1]+crop_size+1, img_y), # height (top:bottom)
                max(position[0]-crop_size, 0):min(position[0]+crop_size+1, img_x) # width (left:right)
                ]
    return gray_img

def draw_circle(img: List, thres_img: List, position: List) -> List:
    """ Blob detect neuron and draw circle around

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
imgs = np.load(f"{video_num}_crop.nd2.npy")
ava = np.load(f"AVA_{video_num}.mat.npy")
avb = np.load(f"AVB_{video_num}.mat.npy")

start_frame = 0
end_frame = 5 # imgs.shape[0]
print(f"{end_frame-start_frame} frames")

(img_y, img_x) = imgs[0, 0, :, :].shape # dimensions
channel = 1
crop_size = 6
thres = 50
suffix = ""

crop = True # to skip cropping, set to False
blob = False # to skip blob detection, set to False
save = True # to save as npy file, set to True

# Crop Images
if crop:
    print("Start Cropping")
    start_crop = time.time()
    try:
        i = 0
        for frame in range(start_frame, end_frame):

            # neuron positions
            position_a = get_frame_coords(frame, ava)
            position_b= get_frame_coords(frame, avb)
            
            # # crop image & save
            # img_a = crop_image(imgs, frame, channel, position_a, crop_size)
            # img_b = crop_image(imgs, frame, channel, position_b, crop_size)
            
            # plt.imsave(f'{file_dir}/cropped_imgs/{video_num}/{frame}a{suffix}.png', img_a, cmap="gray")
            # plt.imsave(f'{file_dir}/cropped_imgs/{video_num}/{frame}b{suffix}.png', img_b, cmap="gray") 
                   
            # print(f"------- cropped frame {frame}-------")
            
            # save original image (to check manually against final product)
            plt.imsave(f'{file_dir}/original/{video_num}/{frame}{suffix}.png', imgs[frame,0,:,:])
    except Exception as e:
        print(f"error cropping frame {frame}")
        print(e)
        pass
    print("End Cropping")
    print(f"Crop Time for {end_frame} frames: {time.time()-start_crop}")


# Threshold Image
if blob:
    print("Start Blob Detection")
    start_blob = time.time()
    npy_imgs=[]
    for frame in range(start_frame, end_frame):
        # Read images        
        img_gray_a = cv2.imread(f"{file_dir}/cropped_imgs/{video_num}/{frame}a{suffix}.png")
        img_gray_b = cv2.imread(f"{file_dir}/cropped_imgs/{video_num}/{frame}b{suffix}.png") 
        
        # Threshold image
        ret_a, thres_img_a = cv2.threshold(img_gray_a, thres, 255, cv2.THRESH_BINARY)
        ret_b, thres_img_b = cv2.threshold(img_gray_b, thres, 255, cv2.THRESH_BINARY)     
        
        # Save threshold images
        plt.imsave(f'{file_dir}/thres_imgs/{video_num}/{frame}a.png', thres_img_a) # save threshold image
        plt.imsave(f'{file_dir}/thres_imgs/{video_num}/{frame}b.png', thres_img_b)
        
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
        plt.imsave(f'{file_dir}/masked/{video_num}/{frame}{suffix}.png', norm_full_img )
        npy_imgs.append(norm_full_img)
        print(f"-------finished frame {frame}-------")
    if save:
        imgset=np.array(npy_imgs)
        np.save(f"allimgs_{video_num}.npy",imgset)       
    print("Finished!")
    print(f"Threshold Time for {end_frame} frames: {time.time()-start_blob}")
    # print(f"Total time: {time.time()-start_crop}")

    
# Check Image