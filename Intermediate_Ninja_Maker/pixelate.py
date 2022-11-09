import cv2
import matplotlib.pyplot as plt
import skimage
from sklearn.cluster import KMeans
from numpy import linalg as lin
import numpy as np
#this is desired image for being pixelated 120 65
img = cv2.imread("/Users/nopparuj/rename this later/IMG/zutomayo.jpg")
#img = cv2.imread("D:\\GIT\\Pixelar-Visualizar\\opencv_pixel\\cassette.jpg")
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

"""
plt.subplot(1,2,1)
plt.imshow(img) #BGR version
plt.subplot(1,2,2)
plt.imshow(rgb) #RGB version
plt.show() 
"""
#Use open cv resize function by shrinking then resize again to original size see 'resize-cv.py'

def pixelate(rgb,w,h):
    height,width = rgb.shape[:2]
    #shrink image to 32x32 px 
    shrink = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
    #return to original size
    return cv2.resize(shrink,(width,height), interpolation=cv2.INTER_NEAREST)

pixel64 = pixelate(rgb,120,65)
pixel164rgb = cv2.cvtColor(pixel64,cv2.COLOR_BGR2RGB)

plt.subplot(1,2,1)
plt.imshow(rgb)
plt.subplot(1,2,2)
plt.imshow(pixel164rgb)
plt.show()
#check unrefined.png
#the result has imperfections that normal pixel art using module 'pixelate' does not .
#K-Means clustering can be used, which to reduce the number of colors in the image and eliminate noise.
#aka group same color together, group same noise togetherm to make result smooth

#Well, I don't understand a damn thing in it
def cluster(idx, img, k):
    clusterValues = []
    for _ in range(0,k):
        clusterValues.append([])
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])
    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]
            
    return imgC

def segment(img,k):
    imgC = np.copy(img)
    
    h = img.shape[0]
    w = img.shape[1]
    
    imgC.shape = (img.shape[0] * img.shape[1], 3)
    
    #5Run k-means on the vectorized responses X to get a vector of labels (the clusters); 
     
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_
    
    # Reshape the label results of k-means so that it has the same size as the input image
    # Return the label image which we call idx
    kmeans.shape = (h, w)

    return kmeans

def KMEAN(image,k):
    idx = segment(image,k)
    return cluster(idx,image,k)


#change pixel and k until image seem satisfied
final= KMEAN(pixel64,5)
finalrgb = cv2.cvtColor(final,cv2.COLOR_BGR2RGB)
plt.imshow(finalrgb)
plt.show()