import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import nd2
from scipy import ndimage
os.chdir("C:/Users/roryg/Desktop/Zhen Lab/CTracking/data")


#img = cv2.imreadmulti("temp3 1.tif")
#total_frames = len(img[1])

img = nd2.imread("11401.nd2")
total_frames = img.shape[0]
locs = np.zeros((total_frames,2))

locs[93,:] = np.array([273,540]) #initial position
for frame_num in range(94,total_frames):
    #set threshold to be 400 for channel 1
    #set threshold to be 300 for channel 0
    channel0 = img[frame_num,0,:,:].copy()
    channel1 = img[frame_num,1,:,:].copy()
    channel0[channel0<300] = 0
    channel0[channel0>0] = 1
    channel1[channel1<400] = 0
    channel1[channel1>0] = 1


    connectivity = 4  #4-connectivity
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(channel0.astype(np.uint8), connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    #calculate min distance from prev identified
    dist_arr = centroids - locs[frame_num-1,:]
    target_component = np.argmin(np.linalg.norm(dist_arr,axis = 1))
    print(centroids[target_component])
    #break
    locs[frame_num,:] = centroids[target_component]
    mask = np.zeros(img.shape[2:])
    mask[output == target_component] = 255
    res = output.copy()
    img[frame_num,0,:,:] = img[frame_num,0,:,:].astype(np.uint8)
    color_img = cv2.cvtColor(img[frame_num,0,:,:], cv2.COLOR_GRAY2BGR)
    cv2.circle(color_img, tuple(centroids[target_component].astype(np.uint32)), 20,(0,255,0))
    #plt.imshow(img[frame_num,0,:,:])
    #plt.show()
    cv2.imwrite("mask"+str(frame_num)+".png", mask)
    #cv2.imwrite(str(frame_num)+".png", img[frame_num,0,:,:].astype(np.uint8))
    #t = np.zeros(image.shape)
    #t[output == np.argmax(sizes)+1] = 255
    #t = t.astype(np.uint8)




