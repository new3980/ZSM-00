import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.imread('/Users/nopparuj/rename this later/IMG/fifth.jpeg',0)

a_low = min(img.flatten())
a_high = max(img.flatten())
print("low=",a_low)
print("high=",a_high)
a_max = 255
a_min = 0

def contrast_stretch(image):

    res = np.zeros((image.shape[0],image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res[i][j] = a_min + ((image[i][j]-a_low)) * ((a_max-a_min)/(a_high-a_low))
            # print(res)
            # print(img)
    
    return res.astype(np.uint8)
    
res = contrast_stretch(img)

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
plt.title("Contrast Stretching")
convert_color(res)
plt.subplot(224)
plt.title("Contrast Stretching")
plt.plot(hist2)
plt.show()

