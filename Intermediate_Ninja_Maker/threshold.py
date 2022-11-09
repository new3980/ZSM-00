import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.imread('/Users/nopparuj/rename this later/IMG/fifth.jpeg',0)
array = img.flatten()
threshold = np.quantile(array,0.5)
print("Threshold = ",threshold)

def threshold_operator(img):
    res = np.zeros((img.shape[0],img.shape[1]))
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            
            if (img[i][j] < threshold):
                res[i][j] = 0
            elif (img[i][j] >= threshold):
                res[i][j] = 255
    
    return res.astype(np.uint8)

res = threshold_operator(img)

hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([res], [0], None, [256], [0, 256])

cv2.imshow('Hi',res)
cv2.waitKey(0)


plt.figure(1)
plt.subplot(221)
plt.title("Original")
convert_color(img)
plt.subplot(222)
plt.title("Original")
plt.plot(hist1)
plt.subplot(223)
plt.title("Thresholding")
convert_color(res)
plt.subplot(224)
plt.title("Thresholding")
plt.plot(hist2)
plt.show()