import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("aau-city-2.jpg", 0)#Make sure image is grayscale
cv2.imshow('image1', img)
#cv2.imshow('image', img)


def GaussFilter(img):
    dims = img.shape
    x = dims[0] #Columns
    y = dims[1] #Rows
    kernelSize = 3 
    offset = kernelSize//2 
    gaussKernel= np.array([[1,2,1],
                  [2,4,2],
                  [1,2,1]])
       
       
    for j in range (offset, y-offset): #Loop through every pixel in the image, the offset is used to avoid the edges of the image
        for i in range (offset, x-offset):
            newValue = 0.0
            
            for l in range(0, kernelSize):#Loop through the kernels rows and columns
                for k in range(0, kernelSize):
                    newValue += img[i+k-offset][j+l -offset]*gaussKernel[l][k] 
  
            newValue=newValue//(16)#16 is sum of the kernel
            img[i][j] = newValue #Set the pixel to the new value calculated for it
            
    return img;

img2 = GaussFilter(img)
cv2.imshow('image2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()           