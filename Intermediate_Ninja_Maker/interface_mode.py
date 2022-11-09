import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

def created_by_nopparuj():
    print("   __  ___     __    _           ")
    print("  /  |/  /__ _/ /__ (_)_ _  ___ _")
    print(" / /|_/ / _ `/  '_// /  ' \/ _ `/")
    print("/_/  /_/\_,_/_/\_\/_/_/_/_/\_,_/ ")
    print("\n<--- By Nopparuj J. 6213460 --->")
    print("\n❄ This is a function for filtering an image ❄ >>")

img = cv2.resize (cv2.imread("/Users/nopparuj/makima_tsuki/IMG/lucy.png",0), (500,500))

# Use with weight = sum of all value in the matrix
block3x3 = np.multiply(np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
]), (1/6))
block5x5 = np.multiply(np.array([
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1]
]), (1/25))
block7x7 = np.multiply(np.array([
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1]
]), (1/49))

# Use with weight = sum of all value in the matrix
gaussian3x3 = np.multiply(np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
]), (1/16))
gaussian5x5 = np.multiply(np.array([
    [1,4,7,4,1],
    [4,16,26,16,4],
    [7,26,41,26,7],
    [4,16,26,16,4],
    [1,4,7,4,1]
]), (1/273))
gaussian7x7 = np.multiply(np.array([
    [0,0,1,2,1,0,0],
    [0,3,13,22,13,3,0],
    [1,13,59,97,59,13,1],
    [2,22,97,159,97,22,2],
    [1,13,59,97,59,13,1],
    [0,3,13,22,13,3,0],
    [0,0,1,2,1,0,0]
]), (1/1003))

# Use with weight = 1
mexican3x3 = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])
mexican5x5 = np.array([
    [0,0,-1,0,0],
    [0,-1,-2,-1,0],
    [-1,-2,16,-2,-1],
    [0,-1,-2,-1,0],
    [0,0,-1,0,0]
])
mexican7x7 = np.array([
    [0,0,-1,-1,-1,0,0],
    [0,-1,-3,-3,-3,-1,0],
    [-1,-3,0,7,0,-3,-1],
    [-1,-3,7,24,7,-3,-1],
    [-1,-3,0,7,0,-3,-1],
    [0,-1,-3,-3,-3,-1,0],
    [0,0,-1,-1,-1,0,0]
])

# From Nout =((Nin + (2*padding) - kernelsize)/stide size) + 1
def findsize(inputsize,kernelsize):
    outputsize = ((inputsize + (2*0) - kernelsize)) + 1
    return outputsize 

def apply_convolution(image,inputkernel,sizeselected,resolution):
    convoluted = np.zeros((sizeselected,sizeselected))

    for u in range (sizeselected):
        for v in range (sizeselected):
            res = (inputkernel * image[u:u+resolution, v:v+resolution]).sum()
            if (res < 0):
                res = 0
            convoluted[u][v] = res
    return convoluted.astype(np.uint8)

def block_compare(original,blockker1,blockker2,blockker3):
    block_filtered1 = apply_convolution(original,blockker1,sizeone,3)
    block_filtered2 = apply_convolution(original,blockker2,sizetwo,5)
    block_filtered3 = apply_convolution(original,blockker3,sizethree,7)
    plt.subplot(2,2,1)
    plt.title("Original")
    convert_color(img)
    plt.subplot(2,2,2)
    plt.title("3x3")
    convert_color(block_filtered1)
    plt.subplot(2,2,3)
    plt.title("5x5")
    convert_color(block_filtered2)
    plt.subplot(2,2,4)
    plt.title("7x7")
    convert_color(block_filtered3)
    plt.show()

def gaussian_compare(original,gauss1,gauss2,gauss3):
    gaussian_filtered1 = apply_convolution(original,gauss1,sizeone,3)
    gaussian_filtered2 = apply_convolution(original,gauss2,sizetwo,5)
    gaussian_filtered3 = apply_convolution(original,gauss3,sizethree,7)
    plt.subplot(2,2,1)
    plt.title("Original")
    convert_color(img)
    plt.subplot(2,2,2)
    plt.title("3x3")
    convert_color(gaussian_filtered1)
    plt.subplot(2,2,3)
    plt.title("5x5")
    convert_color(gaussian_filtered2)
    plt.subplot(2,2,4)
    plt.title("7x7")
    convert_color(gaussian_filtered3)
    plt.show()

def mexican_compare(original,mexi1,mexi2,mexi3):
    mexican_filtered1 = apply_convolution(original,mexi1,sizeone,3)
    mexican_filtered2 = apply_convolution(original,mexi2,sizetwo,5)
    mexican_filtered3 = apply_convolution(original,mexi3,sizethree,7)
    plt.subplot(2,2,1)
    plt.title("Original")
    convert_color(img)
    plt.subplot(2,2,2)
    plt.title("3x3")
    convert_color(mexican_filtered1)
    plt.subplot(2,2,3)
    plt.title("5x5")
    convert_color(mexican_filtered2)
    plt.subplot(2,2,4)
    plt.title("7x7")
    convert_color(mexican_filtered3)
    plt.show()

sizeone = findsize(img.shape[0],3)
sizetwo = findsize(img.shape[0],5)
sizethree = findsize(img.shape[0],7)

# print(sizeone)
# print(sizetwo)
# print(sizethree)

while True:
    created_by_nopparuj()
    print("\n★ Select [1]: To compare Block Filter")
    print("✿ Select [2]: To compare Gaussian Filter")
    print("♥ Select [3]: To compare Mexican Hat Filter")
    print("♣ Select [4] or other integers: To use your own defined filter")

    case = int(input("\nSelect your purpose >>"))
    if (int(case) == 1):
        print("\n ♥ You are now comparing the effect of kernel sizes using the Block Filter")
        block_compare(img,block3x3,block5x5,block7x7)
        quit()
    if (int(case) == 2):
        print("\n ♥ You are now comparing the effect of kernel sizes using the Gaussian Filter")
        gaussian_compare(img,gaussian3x3,gaussian5x5,gaussian7x7)
        quit()
    if (int(case) == 3):
        print("\n ♥ You are now comparing the effect of kernel sizes using the Mexican Hat Filter")
        mexican_compare(img,mexican3x3,mexican5x5,mexican7x7)
        quit()
    if (int(case) == 4):
        print("\n ✤ Now you need to input your own kernel")
        break
    else:
        break

while True:
    rows = int(input("\nEnter kernel rows >>"))
    columns = int(input("Enter kernel columns >>"))

    if (int(rows) != int(columns)):
        print("Please make sure row is equal to column!")

    else:
        break

pre_kernel = []
print("\nEnter the kernel element one by one (row)")
for i in range(rows):
    mat = []
    for j in range(columns):
        mat.append(int(input()))
    pre_kernel.append(mat)

weight = int(input("\nInput corresponding weight (sum of numbers of kernel except for Mexican):"))
kernel = np.multiply(pre_kernel,(1/weight))

defined_size = findsize(img.shape[0],rows)

def user_define(image,inputkernel):
    user_convoluted = np.zeros((defined_size,defined_size))

    for q in range (defined_size):
        for r in range (defined_size):
            user_result = (inputkernel * image[q:q+rows, r:r+columns]).sum()
            if (user_result < 0):
                user_result = 0
            user_convoluted[q][r] = user_result
    return user_convoluted.astype(np.uint8)

user_output = user_define(img,kernel)
# print(img)
# print(user_output)
plt.subplot(1,2,1)
plt.title("Input")
convert_color(img)
plt.subplot(1,2,2)
plt.title("Output")
convert_color(user_output)
plt.show()


