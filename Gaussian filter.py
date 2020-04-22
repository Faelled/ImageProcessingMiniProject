import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("SolaireMem.PNG", 1)
img2 = img.copy()
#cv2.imshow('image', img)

mean = [1,1,1,1,1,1,1,1,1]

def GaussFilter(kernelSize, img):
    dims = img.shape
    x = dims[0]
    y = dims[1]
    offset = kernelSize//2
    customFilter = np.zeros(shape=(kernelSize,kernelSize))
       
    for i in range (0, kernelSize):
        for j in range (0, kernelSize):
            customFilter[i][j] = 1
       
       
            for j in range (offset, y-offset):
                for i in range (offset, x-offset):
                    newValue = 0.0
                    
                    for l in range(0, kernelSize):
                        for k in range(0, kernelSize):
                            newValue += img[i+k-offset][j+l -offset]*customFilter[l][k]
      
                    newValue=newValue//(kernelSize*kernelSize)
                    img[i][j] = newValue
    return img;


cv2.imshow('image2', img2)
cv2.imshow('image1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()           