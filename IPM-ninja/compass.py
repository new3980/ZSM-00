import numpy as np
import cv2
import matplotlib.pyplot as plt

# Compass operator 
def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.resize(cv2.imread("/Users/nopparuj/ZSM-00/IPM-ninja/zuto.png",0), (1000,1000))

robin0 = np.multiply(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), (1/8))
robin1 = np.multiply(np.array([[-2,-1,0],[-1,0,1],[0,1,2]]), (1/8))
robin2 = np.multiply(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]), (1/8))
robin3 = np.multiply(np.array([[0,-1,-2],[1,0,-1],[2,1,0]]), (1/8))
robin4 = np.multiply(np.array([[1,0,-1],[2,0,-2],[1,0,-1]]), (1/8))
robin5 = np.multiply(np.array([[2,1,0],[1,0,-1],[0,-1,-2]]), (1/8))
robin6 = np.multiply(np.array([[1,2,1],[0,0,0],[-1,-2,-1]]), (1/8))
robin7 = np.multiply(np.array([[0,1,2],[-1,0,1],[-2,-1,0]]), (1/8))

def sizing(image,filters):
    out_size = (image.shape[0] - filters.shape[0]) + 1
    return out_size

def apply_filter(image,kernel,outputsize):
    filtered = np.zeros((outputsize,outputsize))
    for i in range (outputsize):
        for j in range (outputsize):
            pre_res = (kernel * image[i:i+3, j:j+3]).sum()
            if (pre_res < 50):
                pre_res = 0
            else:
                pre_res = 255
            filtered[i][j] = pre_res
    return filtered.astype(np.uint8)

def compass_filter(input,kernel0,kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7):

    results = np.zeros((resolution,resolution))
    compass0 = apply_filter(input,kernel0,resolution)
    compass1 = apply_filter(input,kernel1,resolution)
    compass2 = apply_filter(input,kernel2,resolution)
    compass3 = apply_filter(input,kernel3,resolution)
    compass4 = apply_filter(input,kernel4,resolution)
    compass5 = apply_filter(input,kernel5,resolution)
    compass6 = apply_filter(input,kernel6,resolution)
    compass7 = apply_filter(input,kernel7,resolution)

    for u in range (resolution):
        for v in range (resolution):
            pre_out = np.array([compass0[u][v],compass1[u][v],compass2[u][v],compass3[u][v],compass4[u][v],compass5[u][v],compass6[u][v],compass7[u][v]])
            output = pre_out.max()
            if (output < 50):
                output = 0
            else:
                output = 255
            results[u][v] = output
    
    return results.astype(np.uint8)

resolution = sizing(img,robin0) #All sizes are the same
compass_res = compass_filter(img,robin0,robin1,robin2,robin3,robin4,robin5,robin6,robin7)

plt.figure(1)
plt.subplot(1,2,1)
plt.title("Original image")
convert_color(img)
plt.subplot(1,2,2)
plt.title("Compass operator")
convert_color(compass_res)
plt.show()














