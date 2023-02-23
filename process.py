import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import nd2
from scipy import ndimage
os.chdir("C:/Users/roryg/Desktop/Zhen Lab/CTracking/data")
 #Change to your path


#img = cv2.imreadmulti("temp3 1.tif")
#total_frames = len(img[1])

img = nd2.imread("11401.nd2") #Img is an array of size (frames, channels, height, width)
#To access first image we index it by: img[first_frame_num, first_channel, :,:]
#Practice numpy indexing if you are unfamiliar.

plt.imshow(img[0,0,:,:]) #Visualize the first image
plt.show()

total_frames = img.shape[0]
locs = np.zeros((total_frames,2)) #Empty array to store locations of particles






locs[0,:] = np.array([286,225]) #initial position (x,y) for frame 0
for frame_num in range(0,100): #Loop over frames 0-100
    #set threshold to be 400 for channel 1
    #set threshold to be 300 for channel 0
    channel0 = img[frame_num,0,:,:].copy() #Red channel
    channel1 = img[frame_num,1,:,:].copy() #Green channel

    channel0_thresh = channel0.copy()
    channel1_thresh = channel1.copy()

    channel0_thresh[channel0<300] = 0 #threshold red channel
    channel0_thresh[channel0>0] = 1

    channel1_thresh[channel1<400] = 0 #threshold green channel
    channel1_thresh[channel1>0] = 1


    connectivity = 4  #4-connectivity
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(channel0_thresh.astype(np.uint8), connectivity, cv2.CV_32S) #Get all connected components in red channel
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    dist_arr = centroids - locs[frame_num-1,:]  #Calculate distance of identified objects from our previously known location of neuron
    target_component = np.argmin(np.linalg.norm(dist_arr,axis = 1)) #Choose the component with minimum distance


    locs[frame_num,:] = centroids[target_component]
    mask = np.zeros(img.shape[2:])
    mask[output == target_component] = 255
    res = output.copy()
    channel0 = channel0.astype(np.uint8)
    color_img = cv2.cvtColor(channel0, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_img, tuple(centroids[target_component].astype(np.uint32)), 20,(0,255,0)) #Draw a circle around the identified area

    channel0 = np.round(img[frame_num,0,:,:] / img[frame_num,0,:,:].max() * 255).astype(np.uint8)
    channel1 = np.round(img[frame_num,1,:,:] / img[frame_num,0,:,:].max() * 255).astype(np.uint8)
    channel0 = np.concatenate((np.expand_dims(channel0, axis = 2), np.expand_dims(channel1, axis = 2)), axis = 2)

    plt.imshow(mask)
    plt.show()

    #np.save(str(frame_num)+"and2channels.npy", channel0)



    #t = np.zeros(image.shape)
    #t[output == np.argmax(sizes)+1] = 255
    #t = t.astype(np.uint8)





