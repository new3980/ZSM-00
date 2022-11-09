import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.imread('/Users/nopparuj/rename this later/IMG/fifth.jpeg',0)

array = img.flatten()
quantile1 = np.quantile(array,0.005)
# quantile2 = np.quantile(array,0.5)
quantile3 = np.quantile(array,0.995)
# quantile4 = np.quantile(array,1.0)
print(quantile1)
# print(quantile2)
print(quantile3)
# print(quantile4)

def modified(img):
    res = np.zeros((img.shape[0],img.shape[1]))
    amax = 255
    amin = 0
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):

            if (img[i][j] <= quantile1):
                res[i][j] = amin

            if ((quantile1 < img[i][j]) and (img[i][j] < quantile3)):
                res[i][j] = amin + (img[i][j]-quantile1) * ((amax-amin)/(quantile3-quantile1))
            
            if (img[i][j] >= quantile3):
                res[i][j] = amax

    return res.astype(np.uint8)

res = modified(img)

hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([res], [0], None, [256], [0, 256])


plt.figure(1)
plt.subplot(221)
plt.title("Original")
convert_color(img)
plt.subplot(222)
plt.title("Original")
plt.plot(hist1)
plt.subplot(223)
plt.title("Modified Contrast Stretching")
convert_color(res)
plt.subplot(224)
plt.title("Modified Contrast Stretching")
plt.plot(hist2)
plt.show()
            
