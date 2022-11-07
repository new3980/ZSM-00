import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.resize(cv2.imread("C:\\Users\\MBComputer\\Downloads\\powercat.png",0), (1000,1000))

def plot_images(image1,image2,image3,image4):
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.title("Original image")
    convert_color(image1)
    plt.subplot(2,2,2)
    plt.title("Lower = 50 Upper = 150")
    convert_color(image2)
    plt.subplot(2,2,3)
    plt.title("Lower = 10 Upper = 50")
    convert_color(image3)
    plt.subplot(2,2,4)
    plt.title("Lower = 150 Upper = 200")
    convert_color(image4)
    plt.show()



def apply_canny(image,t_upper,t_lower):
    canny_edges = cv2.Canny(image,t_upper,t_lower)
    return canny_edges.astype(np.uint8)

result1 = apply_canny(img,50,150)
result2 = apply_canny(img,10,50)
result3 = apply_canny(img,150,200)

plot_images(img,result1,result2,result3)


