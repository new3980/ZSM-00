import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convert(BGR):
    converted = cv2.cvtColor(BGR,cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.imread('/Users/nopparuj/rename this later/IMG/anya.jpeg')

def bgr2xyz(channel):
    for i in range(len(channel)):
        for j in range(len(channel[i])):
            if (channel[i][j] > 0.04045):
                channel[i][j] = ((channel[i][j]+0.055)/1.055)**2.4
            else:
                channel[i][j] = channel[i][j]/12.92
    return channel

def xyz2lab(channel):
    for i in range(len(channel)):
        for j in range(len(channel[i])):
            if (channel[i][j] > 0.008856):
                channel[i][j] = channel[i][j]**(1/3)
            else: 
                channel[i][j] = (7.787 * channel[i][j]) + (16/166)
    return channel

b = np.array(img[:,:,0]/255)
g = np.array(img[:,:,1]/255)
r = np.array(img[:,:,2]/255)

b_trans = bgr2xyz(b)
g_trans = bgr2xyz(g)
r_trans = bgr2xyz(r)

r = r_trans*100
g = g_trans*100
b = b_trans*100

x = (r_trans*0.4124) + (g_trans*0.3576) + (b_trans*0.1805)
y = (r_trans*0.2126) + (g_trans*0.7152) + (b_trans*0.0722)
z = (r_trans*0.0193) + (g_trans*0.1192) + (b_trans*0.9505)

x_trans = xyz2lab(x)
y_trans = xyz2lab(y)
z_trans = xyz2lab(z)

L = (116*y_trans) - 16
A = 500*(x_trans-y_trans)
B = 200*(y_trans-z_trans)

# checking the L A B values
print(L)
print(A)
print(B)

result = (np.dstack((L,A,B))).astype(np.uint8)

new = Image.fromarray(result, mode='LAB')
new.save('LAB.tiff')

cv2.imshow('LAB', result)

# cv2.imshow('pls',res)
cv2.waitKey(0)
