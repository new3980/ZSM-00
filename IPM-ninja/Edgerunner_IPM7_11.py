import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings


def created_by_nopparuj():
    print("   __  ___     __    _           ")
    print("  /  |/  /__ _/ /__ (_)_ _  ___ _")
    print(" / /|_/ / _ `/  '_// /  ' \/ _ `/")
    print("/_/  /_/\_,_/_/\_\/_/_/_/_/\_,_/ ")
    print("\n<--- By Nopparuj J. 6213460 --->")
    print("\n❄ This python program is able to perform both edge sharpening and unsharp masking ❄")

# Convert BGR to RGB for matplotlib functions
def convert_color(BGR):
    converted = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.resize(cv2.imread("/Users/nopparuj/ZSM-00/IPM-ninja/powercar.png",0), (1000,1000))
warnings.filterwarnings('ignore')

# Required filters
# 1) Laplacian filter
laplacian = np.multiply(np.array([[0,1,0],[1,-4,1],[0,1,0]]), (1/8))
# 2) Gaussian filter
gaussian = np.multiply(np.array([[1,2,1],[2,4,2],[1,2,1]]), (1/16))

# Find outputsize after convolution with filters
def sizing(image,filters):
    out_size = (image.shape[0] - filters.shape[0]) + 1
    return out_size

# Convolution function: result in lesser pixels
def apply_filter(image,kernel):
    outputsize = sizing(image,kernel)
    img_filtered = np.zeros((outputsize,outputsize))
    for i in range (outputsize):
        for j in range (outputsize):
            pre_res = (kernel * image[i:i+kernel.shape[0], j:j+kernel.shape[1]]).sum()
            img_filtered[i][j] = pre_res
    return img_filtered.astype(np.uint8)

# Edge sharpening function with edge padding
def edge_sharpening(input,kernel_filter):
    results = np.zeros((input.shape[0],input.shape[1]))
    post_laplace = apply_filter(input,kernel_filter)
    post_laplace_padded = np.multiply(np.pad(post_laplace, pad_width =1, mode ='edge'),1)
    for u in range (input.shape[0]):
        for v in range (input.shape[1]):
            # Use sharpening factor = 1
            sharped = input[u][v] - post_laplace_padded[u][v]
            if (sharped < 0):
                sharped = 0
            results[u][v] = sharped
    return results.astype(np.uint8)

# Unsharp masking function with edge padding
def unsharp_masking(input,kernel_filter):
    mask = np.zeros((input.shape[0],input.shape[1]))
    unsharped = np.zeros((input.shape[0],input.shape[1]))
    post_gaussian = apply_filter(input,kernel_filter)
    post_gaussian_padded = np.pad(post_gaussian, pad_width =1, mode ='edge')
    for q in range(input.shape[0]):
        for r in range(input.shape[1]):
            pre_mask = input[q][r] - post_gaussian_padded[q][r]
            # Use weight factor = 2 to enhance visibility of the edges
            mask[q][r] = np.multiply(pre_mask,2)
    
            final = input[q][r] + mask[q][r]
            if (final < 0):
                final = 0
            unsharped[q][r] = final
    return unsharped.astype(np.uint8)

# Comparing histogram
def compare_histogram(image_one,image_two):
    hist1 = cv2.calcHist([image_one], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image_two], [0], None, [256], [0, 256])
    return hist1, hist2

def plot_results(image_one,image_two,histogram1,histogram2,my_str1,my_str2,my_str3,my_str4):
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.title(my_str1.title())
    convert_color(image_one)
    plt.subplot(1,2,2)
    plt.title(my_str2.title())
    convert_color(image_two)

    plt.figure(2)
    plt.subplot(1,2,1)
    plt.title(my_str3.title())
    plt.plot(histogram1)
    plt.subplot(1,2,2)
    plt.title(my_str4.title())
    plt.plot(histogram2)

    plt.show()

while True:
    created_by_nopparuj()
    print(">> 1 << : for 'EDGE SHARPENING'")
    print(">> 2 << : for 'UNSHARP MASK'")
    case = int(input("\nSelect yout function >> "))

    if (case == 1):
        print("Applying Edge sharpening")
        edge_sharped = edge_sharpening(img,laplacian)
        before_sharp, after_sharp = compare_histogram(img,edge_sharped)
        plot_results(img,edge_sharped,before_sharp,after_sharp,"Original image","Edge sharpen","Original histogram","Edge sharpen histogram")
        quit()
    
    elif (case == 2):
        print("Applying Unsharp masking")
        unsharp_mask = unsharp_masking(img,gaussian)
        before_mask, after_mask = compare_histogram(img,unsharp_mask)
        plot_results(img,unsharp_mask,before_mask,after_mask,"Original image","Unsharp masked","Original histogram","Unsharp masked histogram")
        quit()
    
    else:
        break





