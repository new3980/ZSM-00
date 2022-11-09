import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

input1 = cv.resize(cv.imread("C:\\Users\\MBComputer\\Downloads\\powercat.png"), (1000,1000))
input2 = cv.resize(cv.imread("C:\\Users\\MBComputer\\Downloads\\ZSMMM\\rika.png"), (600,600))
gray1 = cv.cvtColor(input1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(input2, cv.COLOR_BGR2GRAY)

prewitt_1 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitt_2 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

sobel_1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

robert_1 = np.array([[0,1],[-1,0]])
robert_2 = np.array([[-1,0],[0,1]])

def plot_images(result_one,result_two):
  plt.subplot(121),plt.imshow(result_one,cmap='gray')
  plt.title('Image A')
  plt.xticks([])
  plt.yticks([])

  plt.subplot(122),plt.imshow(result_two,cmap='gray')
  plt.title('Image B')
  plt.xticks([])
  plt.yticks([])
  plt.show()


def apply_weight(kernel,weight_kernel):
  new_kernel = np.multiply(kernel,(1/weight_kernel))
  return new_kernel

def get_edges(image, horizontal_ker, vertical_ker,weight):
  horizontal_kernel = apply_weight(horizontal_ker,weight)
  vertical_kernel = apply_weight(vertical_ker,weight)

  outputsize = (image.shape[0] - horizontal_kernel.shape[0]) + 1 
  results = np.zeros((outputsize,outputsize))
  xaxis = np.zeros((outputsize,outputsize))
  yaxis = np.zeros((outputsize,outputsize))

  for i in range (results.shape[0]):
    for j in range (results.shape[1]):
      pre_vertical = (image[i:i+vertical_kernel.shape[0], j:j+vertical_kernel.shape[1]] * vertical_kernel).sum()
      pre_horizontal = (image[i:i+horizontal_kernel.shape[0], j:j+horizontal_kernel.shape[1]] * horizontal_kernel).sum()

      xaxis[i][j] = pre_horizontal
      yaxis[i][j] = pre_vertical

      pre_output = np.sqrt(((np.power(xaxis[i][j],2)) +(np.power(yaxis[i][j],2))))
      if (pre_output < 5):
        pre_output = 0
      else:
        pre_output = 255
      results[i][j] = pre_output
  return results.astype(np.uint8)

# output = get_edges(gray1,prewitt_1,prewitt_2,6)


while True:
  print("1: Apply Prewitt operator")
  print("2: Apply Sobel operator")
  print("3: Apply Robert operator")
  choice = int(input("\nSelect function >> "))
  if (choice == 1):
    print("\nPrewitt operator")
    prewitt_output1 = get_edges(gray1,prewitt_1,prewitt_2,6)
    prewitt_output2 = get_edges(gray2,prewitt_1,prewitt_2,6)
    plot_images(prewitt_output1,prewitt_output2)
    quit()

  elif (choice == 2):
    print("\nSobel operator")
    sobel_output1 = get_edges(gray1,sobel_1,sobel_2,8)
    sobel_output2 = get_edges(gray2,sobel_1,sobel_2,8)
    plot_images(sobel_output1,sobel_output2)    
    quit()
  
  elif (choice == 3):
    print("\nRobert operator")
    robert_output1 = get_edges(gray1,robert_1,robert_2,2)
    robert_output2 = get_edges(gray2,robert_1,robert_2,2)
    plot_images(robert_output1,robert_output2) 
    quit()
  else:
    break  

