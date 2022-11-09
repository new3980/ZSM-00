import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

salt_n_pepper = cv2.resize (cv2.imread("/Users/nopparuj/makima_tsuki/IMG/try.png",0), (500,500))
img = cv2.resize (cv2.imread("/Users/nopparuj/makima_tsuki/IMG/trythis.png",0), (500,500))


def min_filter(image, outputsize):
    noise_removed = np.zeros((outputsize,outputsize))
    for i in range (outputsize):
        for j in range (outputsize):
            res = image[i:i+3, j:j+3]
            minimum = np.min(res)
            if (minimum < 0):
                minimum = 0
            noise_removed[i][j] = minimum
    return noise_removed.astype(np.uint8)

def median_filter(image, outputsize):
    noise_removed = np.zeros((outputsize,outputsize))
    for i in range (outputsize):
        for j in range (outputsize):
            res = image[i:i+3, j:j+3]
            med = np.median(res)
            if (med < 0):
                med = 0
            noise_removed[i][j] = med
    return noise_removed.astype(np.uint8)

def max_filter(image, outputsize):
    noise_removed = np.zeros((outputsize,outputsize))
    for i in range (outputsize):
        for j in range (outputsize):
            res = image[i:i+3, j:j+3]
            maximum = np.max(res)
            if (maximum < 0):
                maximum = 0
            noise_removed[i][j] = maximum
    return noise_removed.astype(np.uint8)

reduced_min = min_filter(salt_n_pepper,498)
reduced_max = max_filter(salt_n_pepper,498)
reduced_med = median_filter(salt_n_pepper,498)
original_min = min_filter(img,498)
original_max = max_filter(img,498)
original_med = median_filter(img,498)

plt.figure(1)
plt.subplot(1,2,1)
plt.title("Minimum filter (original image)")
convert_color(original_min)
plt.subplot(1,2,2)
plt.title("Minimum filter (with noise)")
convert_color(reduced_min)

plt.figure(2)
plt.subplot(1,2,1)
plt.title("Median filter (original image)")
convert_color(original_med)
plt.subplot(1,2,2)
plt.title("Median filter (with noise)")
convert_color(reduced_med)

plt.figure(3)
plt.subplot(1,2,1)
plt.title("Maximum filter (original image)")
convert_color(original_max)
plt.subplot(1,2,2)
plt.title("Maximum filter (with noise)")
convert_color(reduced_max)

plt.figure(4)
plt.subplot(1,2,1)
plt.title("Original image")
convert_color(img)
plt.subplot(1,2,2)
plt.title("Image with noise")
convert_color(salt_n_pepper)

plt.show()