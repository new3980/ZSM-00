import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convert(BGR):
    converted = cv2.cvtColor(BGR,cv2.COLOR_BGR2RGB)
    plt.imshow(converted)

img = cv2.imread('/Users/nopparuj/rename this later/IMG/view.jpeg')

b = img[:,:,0]/255
g = img[:,:,1]/255
r = img[:,:,2]/255

h, w = img.shape[:2]
H = np.zeros((h,w))
S = np.zeros((h,w))
V = np.zeros((h,w))
cmax = np.zeros((h,w))
cmin = np.zeros((h,w))

for i in range(h):
    for j in range(w):
        cmax[i,j] = max(b[i,j], g[i,j], r[i,j])
        cmin[i,j] = min(b[i,j], g[i,j], r[i,j])
        #V = Cmax
        V[i,j] = cmax[i,j]
        
        #Hue calculation
        if V[i,j] == r[i,j]:
            H[i,j] = 60*(g[i,j]-b[i,j])/(cmax[i,j]-cmin[i,j])
        elif V[i,j]  == g[i,j] :
            H[i,j] = 120 + 60*(b[i,j]-r[i,j])/(cmax[i,j]-cmin[i,j])
        elif V[i,j]  == b[i][j] :
            H[i,j] = 240 + 60*(r[i,j] -g[i,j] )/(cmax[i,j]-cmin[i,j])

        if H[i,j] < 0:
            H[i,j] = H[i,j] + 360
        
        #Saturation calculation
        if V[i,j]  != 0:
            S[i,j] = (V[i,j]-cmin[i,j]) / V[i,j]
        elif V[i,j] == 0:
            S[i,j] = 0


#np.uint8 to change value to 8 bits integer (0 to 255)
H = (H*0.5).astype(dtype=np.uint8)
S = (S*255).astype(dtype=np.uint8)
V = (V*255).astype(dtype=np.uint8)

mergeHSV = cv2.merge((H,S,V))
cvhsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# cv2.imshow('code',mergeHSV)
# cv2.imshow('builtin',cvhsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows

plt.figure(1)
plt.subplot(1,2,1)
plt.title("From manual")
convert(mergeHSV)
plt.subplot(1,2,2)
plt.title("From opencv")
convert(cvhsv)
plt.show()

plt.figure(2)
plt.subplot(2,2,1)
plt.title("HSV domain")
convert(mergeHSV)
plt.subplot(2,2,2)
plt.title("H")
convert(H)
plt.subplot(2,2,3)
plt.title("S")
convert(S)
plt.subplot(2,2,4)
plt.title("V")
convert(V)
plt.show()

