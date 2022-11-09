import cv2
import matplotlib.pyplot as plt
print('Code is running')

#Convert BGR to RGB (for matplotlib)
def convert(BGR):
    converted = cv2.cvtColor(BGR,cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.imread("/Users/nopparuj/rename this later/IMG/ZUTOMAYO.jpg",1)
plt.figure(1)
convert(img)

#Array method
print(img.shape[:])
blue = img[:,:,0]
green = img[:,:,1]
red = img[:,:,2] 

plt.figure(2)
plt.subplot(2,2,1)
plt.title("Original")
convert(img)
plt.subplot(2,2,2)
plt.title("BLUE")
convert(blue)
plt.subplot(2,2,3)
plt.title("GREEN")
convert(green)
plt.subplot(2,2,4)
plt.title("RED")
convert(red)

#Split
(bluesplit, greensplit, redsplit) = cv2.split(img)

plt.figure(3)
plt.subplot(2,2,1)
plt.title("Original")
convert(img)
plt.subplot(2,2,2)
plt.title("BLUE_split")
convert(bluesplit)
plt.subplot(2,2,3)
plt.title("GREEN_split")
convert(greensplit)
plt.subplot(2,2,4)
plt.title("RED_split")
convert(redsplit)


#merge
merge_caveman = cv2.merge([blue, green, red])
merge_split = cv2.merge([bluesplit, greensplit, redsplit])

plt.figure(4)
plt.subplot(1,2,1)
plt.title("Original method")
convert(merge_caveman)
plt.subplot(1,2,2)
plt.title("Split method")
convert(merge_split)
plt.show()
