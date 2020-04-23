# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:29:46 2020

@author: chatr
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import math
import skimage


#greyscale
#1) Take the grayscale of the original image
#reading image 
img = cv2.imread('aau-city-2.jpg',1)
#show original-image, first parameter= name of new window


dims = img.shape #Dimensions of image?
imgWidth = dims[0] # amount of columns
imgHeight = dims[1] # amount of rows
imgChannels = dims[2] # amount of channels; 3 = RGB, 1 = greyscale

windowSize = 50 # size of window in pixels that we move around
step = windowSize // 2 # half (rounded down, just in case) 
window = np.zeros(shape=[windowSize, windowSize, 1], dtype=np.uint8) # make blank image to fill out as window

mean = [1,1,1,1,1,1,1,1,1]

#Converting to grayscale - defining grayscale func.
def ConvertToGreyscale(_img, x, y, z):
    _grey = _img.copy()
    for i in range (0, x):
        for j in range (0, y):
            intensity = 0
            for k in range (z):
                intensity += _img[i][j][k]
            intensity = intensity // z
            _grey[i][j] = intensity
    return _grey


def GaussFilter(kernelSize, img):
    dims = img.shape
    x = dims[0]
    y = dims[1]
    print(x)
    print(y)
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



def SobelOperator(img):
    ##Preallocate the matrices with zeros
    #I = zeroes(size(A))
    I = np.zeros(shape=(img.shape))
    
    #filter masks
    F1 = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    
    F2 = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]
    
    
    #https://www.unioviedo.es/compnum/labs/new/intro_image.html
    #convert the image into a double precision numpy array
    #A = np.asarray(img, dtype= np.float64)
    
    #https://scikit-image.org/docs/dev/api/skimage.html#skimage.img_as_float64
    A= skimage.img_as_float64(img, force_copy=False)
    
    ### do "transformations" and convert back after-> see below ##
    #e.g. use name A1, will be used later, otherwise remember to change below
    
    
#    for i in range (1,imgWidth-2):
#        for j in range (1, imgHeight-2):
#            print(A[i]+2)
#            #Gradient operations
#            #Gx = sum(sum(F1*A(i:i+2, j:j+2)))
#            #Gx = sum(sum(F1*A(([i],[i+2]),([j],[j+2]))))
#            Gx = sum(sum(F1*A[[i]+2,[j]+2]))
#            
#            #GY = sum(sum(F2*A(i:i+2,j:j+2)))
#            #Gy = sum(sum(F2*A(([i],[i+2]),([j],[j+2]))))
#            Gy = sum(sum(F2*A[[i]+2,[j]+2]))
#            
#            #magnitude of vector
#            #I(i+1,j+1) = sqrt(Gx^2+Gy^2)
#            I[i+1, j+1] = math.sqrt(pow(Gx,2)+pow(Gy,2))
    
    for i in range(1,dims[0]-1):
        for j in range(1,dims[1]-1):
            Gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            Gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            I = min(255, np.sqrt(Gx**2 + Gy**2))   
    
            
            
    
    
    
    #As this last matrix range of values differs from  [0,255]  we must fit 
    #again the values into this range with a linear transformation
    #A2 = (A1-np.min(A1))/(np.max(A1)-np.min(A1))*255
    A2 = (I-np.min(I))/(np.max(I)-np.min(I))*255
    
    #convert the type of values in the matrix to unsigned integers of 8 bits
    #A3 = A2.astype(np.uint8)
    A3 = A2.astype(np.uint8)
    
    #Ready to be converted back to an image and saved
    #Im = Image.fromarray(A3)
    #Im.save("edges.jpg")
    
    return A3
            

#Displaying grayscale image    
imgGrey = ConvertToGreyscale(img, imgWidth, imgHeight, imgChannels)
cv2.imshow("Greyscale", imgGrey)
imgGauss = GaussFilter(3,imgGrey)
cv2.imshow("GaussianFilter", imgGauss)

#imgSobel = SobelOperator(imgGauss)
#cv2.imshow("Edges", imgSobel)