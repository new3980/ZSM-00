import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.resize(cv2.imread("D:\\GIT\\ZSM-00\\IPM-ninja\\p1.jpg",0), (1000,1000))
_,img1 = cv2.threshold(img1,80,255,cv2.THRESH_BINARY)
img2 = cv2.resize(cv2.imread("D:\\GIT\\ZSM-00\\IPM-ninja\\p2.jpg",0), (1000,1000))
_,img2 = cv2.threshold(img2,80,255,cv2.THRESH_BINARY)

kernel = np.multiply(np.array([[0,1,0],[1,1,1],[0,1,0]]), 255)

def dilate(image):

    dilated = np.zeros((image.shape[0], image.shape[1]))

    for u in range (1,image.shape[0]-2):
        for v in range (1,image.shape[0]-2):

            if ((image[u-1][v] == kernel[0][1]) or (image[u][v-1] == kernel[1][0]) or (image[u][v] == kernel[1][1]) 
                or (image[u][v+1] == kernel[1][2]) or  (image[u+1][v] == kernel[2][1])):
                dilated[u][v] = 255

            else:
                dilated[u][v] = 0

    return dilated.astype(np.uint8)

def erode(image):
    
    dilated = np.zeros((image.shape[0], image.shape[1]))

    for u in range (1,image.shape[0]-2):
        for v in range (1,image.shape[0]-2):

            if ((image[u-1][v] == kernel[0][1]) and (image[u][v-1] == kernel[1][0]) and (image[u][v] == kernel[1][1]) 
                and (image[u][v+1] == kernel[1][2]) and  (image[u+1][v] == kernel[2][1])):
                dilated[u][v] = 255

            else:
                dilated[u][v] = 0

    return dilated.astype(np.uint8)

dilate_res1 = dilate(img1)
dilate_res2 = dilate(img2)
erode_res1 = erode(img1)
erode_res2 = erode(img2)

#close and open 1 image each
closed = erode(dilate_res1)
opened = dilate(erode_res1)

#Apply erode/dilate to 2 images
plt.figure(1) #image 1
plt.subplot(3,1,1)
plt.title("Pic 1")    
plt.xticks([])
plt.yticks([])
plt.imshow(img1,cmap='gray')
plt.subplot(3,1,2)
plt.title("Dilation (Pic 1)")    
plt.xticks([])
plt.yticks([])
plt.imshow(dilate_res1,cmap='gray')
plt.subplot(3,1,3)
plt.title("Erosion (Pic 1)")    
plt.xticks([])
plt.yticks([])
plt.imshow(erode_res1,cmap='gray')

plt.figure(2) #image 1
plt.subplot(3,1,1)
plt.title("Pic 2")    
plt.xticks([])
plt.yticks([])
plt.imshow(img2,cmap='gray')
plt.subplot(3,1,2)
plt.title("Dilation (Pic 2)")    
plt.xticks([])
plt.yticks([])
plt.imshow(dilate_res2,cmap='gray')
plt.subplot(3,1,3)
plt.title("Erosion (Pic 2)")    
plt.xticks([])
plt.yticks([])
plt.imshow(erode_res2,cmap='gray')

plt.figure(3)
plt.subplot(2,1,1)
plt.title("Closing")    
plt.xticks([])
plt.yticks([])
plt.imshow(closed,cmap='gray')
plt.subplot(2,1,2)
plt.title("Opening")    
plt.xticks([])
plt.yticks([])
plt.imshow(opened,cmap='gray')

plt.show()

