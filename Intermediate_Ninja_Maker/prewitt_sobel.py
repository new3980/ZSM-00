from cv2 import sqrt
import numpy as np
import cv2 
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img_one = cv2.resize (cv2.imread("/Users/nopparuj/makima_tsuki/IMG/makima_one.jpeg",0), (500,500))
img_two = cv2.resize (cv2.imread("/Users/nopparuj/makima_tsuki/IMG/makima_two.jpeg",0), (500,500))
rain = cv2.resize (cv2.imread("/Users/nopparuj/makima_tsuki/IMG/rain.jpeg",0), (500,500))

#Prewitt filters in X and Y
prewitt_x = np.multiply(np.array([[-1,0,1],[-1,0,1],[-1,0,1]]), (1/6))
prewitt_y = np.multiply(np.array([[-1,-1,-1],[0,0,0],[1,1,1]]), (1/6))

#Sobel filters in X and Y
sobel_x = np.multiply(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), (1/8))
sobel_y = np.multiply(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]), (1/8))

def edge_detection(image,kernel1,kernel2,sizeselected):
    convoluted = np.zeros((sizeselected,sizeselected))
    horizontal = np.zeros((sizeselected,sizeselected))
    vertical = np.zeros((sizeselected,sizeselected))

    for i in range (sizeselected):
        for j in range (sizeselected):
            res_x = (kernel1 * image[i:i+3, j:j+3]).sum()
            horizontal[i][j] = res_x

            res_y = (kernel2 * image[i:i+3, j:j+3]).sum()
            vertical[i][j] = res_y

            res = np.sqrt(((np.power(horizontal[i][j],2)) + (np.power(vertical[i][j],2))))
            if (res < 10):
                res = 0
            else:
                res = 255
            convoluted[i][j] = res
    return convoluted.astype(np.uint8), horizontal.astype(np.uint8), vertical.astype(np.uint8)

output_prewitt1, horizontal_prewitt1, vertical_prewitt1 = edge_detection(img_one, prewitt_x, prewitt_y, 498)
output_prewitt2, horizontal_prewitt2, vertical_prewitt2 = edge_detection(img_two, prewitt_x, prewitt_y, 498)
print(output_prewitt1)
print(output_prewitt2)

output_sobel1, horizontal_sobel1, vertical_sobel1 = edge_detection(img_one, sobel_x, sobel_y, 498)
output_sobel2, horizontal_sobel2, vertical_sobel2 = edge_detection(img_two, sobel_x, sobel_y, 498)
print(output_sobel1)
print(output_sobel2)

# rain_prewitt, horizontal_rain1, vertical_rain1 = edge_detection(rain, prewitt_x, prewitt_y, 498)
# rain_sobel, horizontal_rain2, vertical_rain2 = edge_detection(rain, sobel_x, sobel_y, 498)
# plt.subplot(1,2,1)
# plt.title("Prewitt operator")
# convert_color(rain_prewitt)
# plt.subplot(1,2,2)
# plt.title("Sobel operator")
# convert_color(rain_sobel)
# plt.show()

plt.figure(1)
plt.subplot(2,2,1)
plt.title("Input image A")
convert_color(img_one)
plt.subplot(2,2,2)
plt.title("Prewitt: X-axis")
convert_color(horizontal_prewitt1)
plt.subplot(2,2,3)
plt.title("Prewitt: Y-axis")
convert_color(vertical_prewitt1)
plt.subplot(2,2,4)
plt.title("Edged image")
convert_color(output_prewitt1)


plt.figure(2)
plt.subplot(2,2,1)
plt.title("Input image B")
convert_color(img_two)
plt.subplot(2,2,2)
plt.title("Prewitt: X-axis")
convert_color(horizontal_prewitt2)
plt.subplot(2,2,3)
plt.title("Prewitt: Y-axis")
convert_color(vertical_prewitt2)
plt.subplot(2,2,4)
plt.title("Edged image")
convert_color(output_prewitt2)

plt.figure(3)
plt.subplot(2,2,1)
plt.title("Input image A")
convert_color(img_one)
plt.subplot(2,2,2)
plt.title("Sobel: X-axis")
convert_color(horizontal_sobel1)
plt.subplot(2,2,3)
plt.title("Sobel: Y-axis")
convert_color(vertical_sobel1)
plt.subplot(2,2,4)
plt.title("Edged image")
convert_color(output_sobel1)

plt.figure(4)
plt.subplot(2,2,1)
plt.title("Input image B")
convert_color(img_two)
plt.subplot(2,2,2)
plt.title("Sobel: X-axis")
convert_color(horizontal_sobel2)
plt.subplot(2,2,3)
plt.title("Sobel: Y-axis")
convert_color(vertical_sobel2)
plt.subplot(2,2,4)
plt.title("Edged image")
convert_color(output_sobel2)

plt.figure(5)
plt.subplot(1,2,1)
plt.title("Image A: Prewitt")
convert_color(output_prewitt1)
plt.subplot(1,2,2)
plt.title("Image B: Prewitt")
convert_color(output_prewitt2)

plt.figure(6)
plt.subplot(1,2,1)
plt.title("Image A: Sobel")
convert_color(output_sobel1)
plt.subplot(1,2,2)
plt.title("Image B: Sobel")
convert_color(output_sobel2)

plt.show()

