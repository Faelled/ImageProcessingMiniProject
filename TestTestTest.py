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
from Gaussianfilter import GaussFilter

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





def SobelOperator(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1][0] + 2*img[i][j - 1][0] + img[i + 1][j - 1][0]) - (img[i - 1][j + 1][0] + 2*img[i][j + 1][0] + img[i + 1][j + 1][0])
            gy = (img[i - 1][j - 1][0] + 2*img[i - 1][j][0] + img[i - 1][j + 1][0]) - (img[i + 1][j - 1][0] + 2*img[i + 1][j][0] + img[i + 1][j + 1][0])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
            
    return container
    pass
            

#def HarrisCorner(img):
#    
#    for i in range(y):
#        print("Looking for them fat corners brah")
#        for j in range(X):
#            
#    
#    return harrisImg
    

#Displaying grayscale image    
imgGrey = ConvertToGreyscale(img, imgWidth, imgHeight, imgChannels)
cv2.imshow("Greyscale", imgGrey)
#Show image with gaussian-filter applied
imgGauss = GaussFilter(imgGrey)
cv2.imshow("GaussianFilter", imgGauss)

#Show image with sobelOperator applied
imgSobel = SobelOperator(imgGauss)
cv2.imshow("Edges", imgSobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
