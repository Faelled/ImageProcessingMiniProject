import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("aau-city-2.jpg", 0)
cv2.imshow('image1', img)
#cv2.imshow('image', img)

mean = [1,1,1,1,1,1,1,1,1]

def GaussFilter(img):
    dims = img.shape
    x = dims[0]
    y = dims[1]
    kernelSize = 3
    offset = kernelSize//2
    gaussKernel= np.array([[1,2,1],
                  [2,4,2],
                  [1,2,1]])
       
       
    for j in range (offset, y-offset):
        for i in range (offset, x-offset):
            newValue = 0.0
            
            for l in range(0, kernelSize):
                for k in range(0, kernelSize):
                    newValue += img[i+k-offset][j+l -offset]*gaussKernel[l][k]
  
            newValue=newValue//(16)
            img[i][j] = newValue
            
    return img;

img2 = GaussFilter(img)
cv2.imshow('image2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()           