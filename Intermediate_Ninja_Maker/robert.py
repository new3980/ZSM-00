from cv2 import sqrt
import numpy as np
import cv2 
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img_one = cv2.resize (cv2.imread("/Users/nopparuj/makima_tsuki/IMG/makima_one.jpeg",0), (500,500))
img_two = cv2.resize (cv2.imread("/Users/nopparuj/makima_tsuki/IMG/makima_two.jpeg",0), (500,500))

#Robert filters in X and Y
robert_one = np.multiply(np.array([[0,1],[-1,0]]), (1/2))
robert_two = np.multiply(np.array([[-1,0],[0,1]]), (1/2))

def robert_edge(image,kernelx,kernely,sizeselected):
    convoluted = np.zeros((sizeselected,sizeselected))
    horizontal = np.zeros((sizeselected,sizeselected))
    vertical = np.zeros((sizeselected,sizeselected))

    for i in range (sizeselected):
        for j in range (sizeselected):
            res_x = (kernelx * image[i:i+2, j:j+2]).sum()
            horizontal[i][j] = res_x

            res_y = (kernely * image[i:i+2, j:j+2]).sum()
            vertical[i][j] = res_y

            res = np.sqrt(((np.power(horizontal[i][j],2)) + (np.power(vertical[i][j],2))))
            if (res < 5):
                res = 0
            else:
                res = 255
            convoluted[i][j] = res
    return convoluted.astype(np.uint8), horizontal.astype(np.uint8), vertical.astype(np.uint8)

output_robert1, horizontal_robert1, vertical_robert1 = robert_edge(img_one, robert_one, robert_two, 498)
output_robert2, horizontal_robert2, vertical_robert2 = robert_edge(img_two, robert_one, robert_two, 498)
print(output_robert1)
print(output_robert2)


plt.figure(1)
plt.subplot(2,2,1)
plt.title("Input image A")
convert_color(img_one)
plt.subplot(2,2,2)
plt.title("Kernel one")
convert_color(horizontal_robert1)
plt.subplot(2,2,3)
plt.title("Kernel two")
convert_color(vertical_robert1)
plt.subplot(2,2,4)
plt.title("Edged image")
convert_color(output_robert1)


plt.figure(2)
plt.subplot(2,2,1)
plt.title("Input image B")
convert_color(img_two)
plt.subplot(2,2,2)
plt.title("Kernel one")
convert_color(horizontal_robert2)
plt.subplot(2,2,3)
plt.title("Kernel two")
convert_color(vertical_robert2)
plt.subplot(2,2,4) 
plt.title("Edged image")
convert_color(output_robert2)

plt.figure(3)
plt.subplot(1,2,1)
plt.title("Image A: Robert")
convert_color(output_robert1)
plt.subplot(1,2,2)
plt.title("Image B: Robert")
convert_color(output_robert2)

plt.show()