import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.resize(cv2.imread("D:\\GIT\\ZSM-00\\IPM-ninja\\asa.png",0), (850,850))
_,img = cv2.threshold(img,50,255,cv2.THRESH_BINARY)

ones = np.array([[255,255,255],[255,255,255],[255,255,255]])

def erosion(image,window):
    padded = np.pad(image, pad_width =1, mode ='edge')
    output = np.zeros((image.shape[0],image.shape[1]))

    for i in range (padded.shape[0]-2):
        for j in range (padded.shape[1]-2):
            if ((padded[i][j+1] == window[0][1]) and (padded[i+1][j+1] == window[1][1]) and
                (padded[i+1][j] == window[1][0]) and (padded[i+1][j+2] == window[1][2]) and 
                (padded[i+2][j+1] == window[2][1]) and (padded[i][j] == window[0][0]) and
                (padded[i+2][j] == window[2][0]) and (padded[i][j+2] == window[0][2]) and
                (padded[i+2][j+2] == window[2][2])):

                output[i][j] = 255
                
            else:
                output[i][j] = 0

    return output

def dilation(image,window):
    padded = np.pad(image, pad_width =1, mode ='edge')
    output = np.zeros((image.shape[0],image.shape[1]))

    for i in range (padded.shape[0]-2):
        for j in range (padded.shape[1]-2):
            if ((padded[i][j+1] == window[0][1]) or (padded[i+1][j+1] == window[1][1]) or
                (padded[i+1][j] == window[1][0]) or (padded[i+1][j+2] == window[1][2]) or 
                (padded[i+2][j+1] == window[2][1]) or (padded[i][j] == window[0][0]) or
                (padded[i+2][j] == window[2][0]) or (padded[i][j+2] == window[0][2]) or
                (padded[i+2][j+2] == window[2][2])):

                output[i][j] = 255

            else:
                output[i][j] = 0

    return output

dilated = dilation(img,ones)
eroded = erosion(img,ones)

opening = dilation(erosion(img,ones),ones)
closing = erosion(dilation(img,ones),ones)

#Using OpenCV
kernel = np.ones((3, 3), np.uint8)
cvdilate = cv2.dilate(img,kernel)
cverode = cv2.erode(img,kernel)
cvopen = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cvclose = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.figure(1)
plt.subplot(1,3,1)
plt.title("OpenCV: Dilation")    
plt.xticks([])
plt.yticks([])
plt.imshow(cvdilate, cmap='gray')
plt.subplot(1,3,2)
plt.title("OpenCV: Erosion")
plt.xticks([])
plt.yticks([])
plt.imshow(cverode, cmap='gray')
plt.subplot(1,3,3)
plt.title("Original Image")
plt.xticks([])
plt.yticks([])
plt.imshow(img, cmap='gray')

plt.figure(2)
plt.subplot(1,3,1)
plt.title("Dilation")
plt.xticks([])
plt.yticks([])
plt.imshow(dilated, cmap='gray')
plt.subplot(1,3,2)
plt.title("Erosion")
plt.xticks([])
plt.yticks([])
plt.imshow(eroded, cmap='gray')
plt.subplot(1,3,3)
plt.title("Original Image")
plt.xticks([])
plt.yticks([])
plt.imshow(img, cmap='gray')

plt.figure(3)
plt.subplot(1,2,1)
plt.title("Opening")
plt.xticks([])
plt.yticks([])
plt.imshow(opening, cmap='gray')
plt.subplot(1,2,2)
plt.title("Closing")
plt.xticks([])
plt.yticks([])
plt.imshow(closing, cmap='gray')

plt.figure(4)
plt.subplot(1,2,1)
plt.title("OpenCV: Opening")
plt.xticks([])
plt.yticks([])
plt.imshow(cvopen, cmap='gray')
plt.subplot(1,2,2)
plt.title("OpenCV: Closing")
plt.xticks([])
plt.yticks([])
plt.imshow(cvclose, cmap='gray')

plt.show()