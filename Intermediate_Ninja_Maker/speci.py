import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(bgr):
    converted_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(converted_image)

img = cv2.imread('/Users/nopparuj/rename this later/IMG/fifth.jpeg',0)
ref = cv2.imread('/Users/nopparuj/rename this later/IMG/rename.png',0)    

def findcdf(hist): #From Alg. 4.1 Pg.68
    k = len(hist) #size of histogram
    n = 0

    for u in range(k):
        n = n + hist[u]

    cdf_hist = np.zeros(k)
    c = hist[0]
    cdf_hist[0] = c/n

    for v in range(k):
        c += hist[v]
        cdf_hist[v] = c/n

    return cdf_hist

def specify(image, reference):
    img_input = cv2.calcHist([image], [0], None, [256], [0,256])
    ref_input = cv2.calcHist([reference], [0], None, [256], [0,256])

    k = len(img_input)

    input_cdf = findcdf(img_input)
    ref_cdf = findcdf(ref_input)

    arraymap = np.zeros(k)

    for u in range(k):
        v = k - 1
        while ((v >= 0) and (ref_cdf[u] <= input_cdf[v])):
            arraymap[u] = v
            v -= 1
    
    return input_cdf, ref_cdf, arraymap

input_cdf, ref_cdf, arraymap = specify(img, ref)

def denormalize_cdf(cdf_forplot):
    forplot = cdf_forplot*255
    plt.plot(forplot)

plt.figure(1)

plt.subplot(3,2,1)
plt.title("Intensity CDF (Original)")
denormalize_cdf(input_cdf)

plt.subplot(3,2,2)
plt.title("Original")
convert_color(img)

plt.subplot(3,2,3)
plt.title("Intensity CDF (Reference)")
denormalize_cdf(ref_cdf)

plt.subplot(3,2,4)
plt.title("Reference")
convert_color(ref)

plt.subplot(3,2,5)
plt.title("Compare CDF")
denormalize_cdf(input_cdf)
denormalize_cdf(ref_cdf)
plt.plot(arraymap)
plt.legend(['Original', 'Reference', 'Result'])

plt.subplot(3,2,6)
plt.title("Result")
specific_map = np.interp(input_cdf, ref_cdf, range(256))
res = (np.reshape(specific_map[img.ravel()], img.shape)).astype(np.uint8)
convert_color(res)

histogramin = cv2.calcHist([img], [0], None, [256], [0,256])
histogramref = cv2.calcHist([ref], [0], None, [256], [0,256])
histogramout = cv2.calcHist([res], [0], None, [256], [0,256])

plt.figure(2)
plt.subplot(3,1,1)
plt.title('Histogram of input image')
plt.plot(histogramin)
plt.subplot(3,1,2)
plt.title('Histogram of reference image')
plt.plot(histogramref)
plt.subplot(3,1,3)
plt.title('Histogram of output image')
plt.plot(histogramout)
plt.show()

