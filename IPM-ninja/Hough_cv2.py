import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math


def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

#Hough line transform (OpenCV-based)
img = cv2.resize(cv2.imread("/Users/nopparuj/ZSM-00/IPM-ninja/room.png",0), (1000,1000))


def hough_linePdetect(image):
    gray = img.copy()
    get_edges = cv2.Canny(image,50,150)
    get_lines = cv2.HoughLinesP(get_edges,1,np.pi/180,threshold=150, minLineLength=10, maxLineGap=100)
    for i in get_lines:
        coor1, coor2, coor3, coor4 = i[0]
        res = cv2.line(gray,(coor1,coor2),(coor3,coor4),(0,0,0),2)
    return res

result = hough_linePdetect(img)

plt.subplot(1,2,1)
plt.title("Original")
convert_color(img)
plt.subplot(1,2,2)
plt.title("Hough transform : Line detection")
convert_color(result)
plt.show()