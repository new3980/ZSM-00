import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.resize (cv2.imread("/Users/nopparuj/rename this later/IMG/lucy.png",0), (500,500))

while True:
    matrixR = int(input("Enter rows:"))
    matrixC = int(input("Enter columns:"))

    if (int(matrixR) != int(matrixC)):
        print("Please make sure row = col!")

    else:
        break

kernel = []

for i in range(matrixR):
    mat = []
    for j in range(matrixC):
        mat.append(int(input()))
    kernel.append(mat)

weight = int(input("Input corresponding weight (sum of numbers of kernel):"))
kernel_new = np.multiply(kernel,(1/weight))

# From Nout =((Nin + (2*padding) - kernelsize)/stide size) + 1
def findsize(inputsize,kernelsize):
    outputsize = ((inputsize + (2*0) - kernelsize)) + 1
    return outputsize 

def apply_convolution(image,inputkernel):
    convoluted = np.zeros((sizes,sizes))
    res = np.zeros((sizes,sizes))

    for u in range (sizes):
        for v in range (sizes):
            res = (inputkernel * image[u:u+matrixR, v:v+matrixC]).sum()
            if (res < 0):
                res = 0
            convoluted[u][v] = res
    return convoluted.astype(np.uint8)

sizes = findsize(img.shape[0],matrixR)
# print(sizes)
# print(kernel_new)
result = apply_convolution(img,kernel_new)
# print("Input",img)
# print("Output",result)

plt.title("7x7")
convert_color(result)
plt.show()

