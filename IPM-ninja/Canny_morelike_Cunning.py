import numpy as np
import cv2
import matplotlib.pyplot as plt

# Try cunny operator without using built-in

# Convert BGR to RGB for matplotlib functions
def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

# Load image in grayscale
img = cv2.resize(cv2.imread("/Users/nopparuj/ZSM-00/IPM-ninja/powercat.png",0), (1000,1000))

# Needed filter
gaussian = np.multiply(np.array([[1,2,1],[2,4,2],[1,2,1]]), (1/16))
sobel_x = np.multiply(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), (1/8))
sobel_y = np.multiply(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]), (1/8))

# Find outputsize after convolution with filters
def sizing(image,filters):
    out_size = (image.shape[0] - filters.shape[0]) + 1
    return out_size

# Normalization
def normal(image):
    image = image / np.max(image)
    return image

# Apply gaussian then perform edge padding (from 998px --> 1000px)
def apply_gaussian(image,g_kernel):
    outputsize = sizing(image,g_kernel)
    smoothed = np.zeros((outputsize,outputsize))
    pre_smoothed = np.zeros((outputsize,outputsize))

    for i in range (outputsize):
        for j in range (outputsize):
            conv = (g_kernel * image[i:i+g_kernel.shape[0], j:j+g_kernel.shape[1]]).sum()
            if (conv < 0):
                conv = 0
            pre_smoothed[i][j] = conv

    smoothed = np.pad(pre_smoothed, pad_width =1, mode ='edge')

    return smoothed.astype(np.uint8)

#Apply general edge detction (Sobel is used)
def general_detect(input_img,xaxis,yaxis):
    smooth_img = apply_gaussian(input_img,gaussian)
    resolution = sizing(smooth_img,xaxis)
    magnitude = np.zeros((resolution,resolution))
    horizontal = np.zeros((resolution,resolution))
    vertical = np.zeros((resolution,resolution))
    theta = np.zeros((resolution,resolution))

    for u in range (resolution):
        for v in range (resolution):
            res_x = (xaxis * smooth_img[u:u+xaxis.shape[0], v:v+xaxis.shape[1]]).sum()
            horizontal[u][v] = res_x
            res_y = (yaxis * smooth_img[u:u+yaxis.shape[0], v:v+yaxis.shape[1]]).sum()
            vertical[u][v] = res_y

            pre_magnitude = np.sqrt(((np.power(horizontal[u][v],2)) +(np.power(vertical[u][v],2))))
            # if (pre_magnitude < 5):
            #     pre_magnitude = 0
            # else:
            #     pre_magnitude = 255
            magnitude[u][v] = pre_magnitude

            theta[u][v] = np.degrees(np.arctan2(vertical[u][v],horizontal[u][v]))
            if (theta[u][v] < 0):
                theta[u][v] += 180

    return magnitude.astype(np.uint8), theta

# Apply Non-Max-Suppression
def nonmax_suppress(magni,dir):
    pre_canny = np.zeros(magni.shape)
    for i in range (dir.shape[0]-1):
        for j in range (dir.shape[1]-1):

            if ((dir[i][j] < 22.5) or (dir[i][j] >= 157.5)):
                if (magni[i][j-1] < magni[i][j] > magni[i][j+1]):
                    pre_canny[i][j] = magni[i][j]
                else:
                    pre_canny[i][j] = 0

            elif ((dir[i][j] >= 22.5) or (dir[i][j] < 67.5)):
                if (magni[i-1][j-1] < magni[i][j] > magni[i+1][j+1]):
                    pre_canny[i][j] = magni[i][j]
                else:
                    pre_canny[i][j] = 0
            
            elif ((dir[i][j] >= 67.55) or (dir[i][j] < 112.5)):
                if (magni[i-1][j] < magni[i][j] > magni[i+1][j]):
                    pre_canny[i][j] = magni[i][j]
                else:
                    pre_canny[i][j] = 0

            elif ((dir[i][j] >= 112.5) or (dir[i][j] < 157.5)):
                if ((magni[i-1][j+1] < magni[i][j] > magni[i+1][j-1])):
                    pre_canny[i][j] = magni[i][j]
                else:
                    pre_canny[i][j] = 0 

    return pre_canny.astype(np.uint8)
                
# Bruh it broke, even I copied thresholding from other

def double_threshold(localmax,low_t,high_t):

    thresholded = np.zeros((localmax.shape[0],localmax.shape[1]))

    for u in range (localmax.shape[0]):
        for v in range (localmax.shape[1]):
            # change value that in the middle between low and high threshold to become constant
            if (low_t <= localmax[u][v] < high_t):
                thresholded[u][v] = 50 # weak edge
            elif (localmax[u][v] < low_t):
                thresholded[u][v] = 0
            elif (localmax[u][v] > high_t):
                thresholded[u][v] = 255 # strong edge
    return thresholded.astype(np.uint8)

def hysteresis(img_thresholded):
    final = np.zeros(img_thresholded.shape)

    for i in range(img_thresholded.shape[0]):		
        for j in range(img_thresholded.shape[1]):
            check = img_thresholded[i][j]

            #check is this pixel is connected to 'strong' edges or not
            if check == 50:			
                if ((img_thresholded[i-1][j] == 255) or (img_thresholded[i+1][j] == 255)
                    or (img_thresholded[i-1][j+1] == 255) or (img_thresholded[i+1][j+1] == 255)
                    or (img_thresholded[i+1][j-1] == 255) or (img_thresholded[i-1][j-1] == 255)
                    or (img_thresholded[i][j+1] == 255) or (img_thresholded[i][j-1] == 255)):

                    final[i][j] = 255		

            elif check == 255:
                final[i][j] = 255

            else:
                final[i][j] = 0		

    return final.astype(np.uint8)

sobel_res, direction = general_detect(img,sobel_x,sobel_y)
local_maximum = nonmax_suppress(sobel_res,direction)
double_thresholded = double_threshold(local_maximum,1,5)
canny = hysteresis(double_thresholded)

plt.figure(1)
plt.subplot(121)
convert_color(sobel_res)
plt.title('After Sobel filter')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
convert_color(canny)
plt.title('Canny performed')
plt.xticks([])
plt.yticks([])

plt.show()
