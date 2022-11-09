import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.imread('/Users/nopparuj/rename this later/IMG/fifth.jpeg',0)

def calcHistogram(image):
    hist_compare = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist_compare

def histogram_plot(image):
    histogram = np.zeros(256)
    for i in image.flatten():
        histogram[i] += 1
    #We can divide the dimension either here or in equalizer(cdf)
    histopdf = histogram#/(img.shape[0]*img.shape[1])
    histocdf = np.cumsum(histopdf)
    return histocdf

histogramcdf = histogram_plot(img) #Histogram of the original image

def equalizer(cdf):
    res = np.zeros((img.shape[0],img.shape[1]))
    for u in range(img.shape[0]):
        for v in range(img.shape[1]):
            #If divide by dimension here, histogram_plot don't need to
            intensity = cdf[img[u][v]]*((255)/(img.shape[0]*img.shape[1]))
            res[u][v] = intensity
    return res.astype(np.uint8)

resolution = equalizer(histogramcdf)
equal_histo = histogram_plot(resolution)
x = range(0,256)

#Histogram check
before = calcHistogram(img)
after = calcHistogram(resolution)

plt.subplot(231)
plt.title('Original')
convert_color(img)
plt.subplot(232)
plt.title('Original CDF')
plt.plot(x,histogramcdf)
plt.subplot(233)
plt.title('Original Histogram')
plt.plot(before)
plt.subplot(234)
plt.title('Equalized')
convert_color(resolution)
plt.subplot(235)
plt.title('Equalized CDF')
plt.plot(x,equal_histo)
plt.subplot(236)
plt.title('Equalized Histogram')
plt.plot(after)
plt.show()


