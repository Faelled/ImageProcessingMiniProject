# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:41:04 2020

@author: chatr
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import system
from skimage.feature import corner_harris, corner_peaks


img = cv2.imread('aau-city-2.jpg',1)


#Variables
dims = img.shape #Dimensions of image?
imgWidth = dims[0] # amount of columns
imgHeight = dims[1] # amount of rows
imgChannels = dims[2] # amount of channels; 3 = RGB, 1 = greyscale

windowSize = 50 # size of window in pixels that we move around
step = windowSize // 2 # half (rounded down, just in case) 
window = np.zeros(shape=[windowSize, windowSize, 1], dtype=np.uint8) # make blank image to fill out as window


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
                    newValue += img[i+k-offset][j+l -offset][0]*gaussKernel[l][k] 
  
            newValue=newValue//(16)#16 is sum of the kernel
            img[i][j] = newValue #Set the pixel to the new value calculated for it
            
    return img;


##Applying the sobel operator
#def SobelOperator(img):
#    container = np.copy(img)
#    size = container.shape
#    for i in range(1, size[0] - 1):
#        for j in range(1, size[1] - 1):
#            gx = (img[i - 1][j - 1][0] + 2*img[i][j - 1][0] + img[i + 1][j - 1][0]) - (img[i - 1][j + 1][0] + 2*img[i][j + 1][0] + img[i + 1][j + 1][0])
#            gy = (img[i - 1][j - 1][0] + 2*img[i - 1][j][0] + img[i - 1][j + 1][0]) - (img[i + 1][j - 1][0] + 2*img[i + 1][j][0] + img[i + 1][j + 1][0])
#            #magnitude of vector ?
#            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
#            
#    #ret,thresh1 = cv2.threshold(img, 140, 255,  cv2.THRESH_BINARY)
##    return thresh1            
#   
#    return container
#   #could compute Non-maximum suppression and/or double thresholding after

def SobelOperator(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0]-1):
        for j in range(1, size[1]-1):
            gx = SobelGradientX(img, i, j)
            gy = SobelGradientY(img, i, j)
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container

def SobelGradientX(img, i, j):
    return (img[i - 1][j - 1][0] + 2*img[i][j - 1][0] + img[i + 1][j - 1][0]) - (img[i - 1][j + 1][0] + 2*img[i][j + 1][0] + img[i + 1][j + 1][0])

def SobelGradientY(img, i, j):
    return (img[i - 1][j - 1][0] + 2*img[i - 1][j][0] + img[i - 1][j + 1][0]) - (img[i + 1][j - 1][0] + 2*img[i + 1][j][0] + img[i + 1][j + 1][0])

def makeWindow(x,y, img):
    row = 0
    col = 0
    
    for i in range (x-step, x+step):
        col = 0
        for j in range (y-step, y+step):
            window[row][col] = img[i][j][0]
            col +=1
        row +=1
                      
def DisplaceWindow(x,y, img):
    for i in range ()

def CalculateWindowR(window):
    k = 0.04
    
    Ix = SobelGradientX(window)
    Iy = SobelGradientY(window)
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    detA = Ixx * Iyy - Ixy * Ixy
    traceA = Ixx + Iyy
    
    harris_response = detA - k * (traceA * traceA)
    
    return harris_response
    #return det - k * (trace * trace) #harris_response


def Harris(img):    #should not be used? is commented out in: https://github.com/muthuspark/ml_research/blob/master/Process%20of%20Harris%20Corner%20Detection%20Algorithm.ipynb
    dims = img.shape #Dimensions of image?
    x = dims[0] # amount of columns
    y = dims[1] # amount of rows
    #height, width = img.shape
    CalculateWindowR.harris_response = []
    window_size = 6
    offset = int(window_size/2)
    k = 0.04
    
    for y in range(offset, y-offset):
        for x in range(offset, x-offset):
            Sxx = np.sum(CalculateWindowR.Ixx[y-offset: y+1+offset, x-offset: x+1+offset])
            Syy = np.sum(CalculateWindowR.Iyy[y-offset: y+1+offset, x-offset: x+1+offset])
            Sxy = np.sum(CalculateWindowR.Ixy[y-offset: y+1+offset, x-offset: x+1+offset])
            
            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy*Sxy)
            trace = Sxx + Syy
            r = det - k* (trace * trace)
            
            CalculateWindowR.harris_response.append(r)
    return r #???

#Non-maximum suppression
def NonMaximumSuppression(img): #should not be a function?
    img_copy_for_corners = np.copy(img)
    img_copy_for_edges = np.copy(img)
    
    for rowIndex, response in enumerate(CalculateWindowR.harris_response):
        for colIndex, r in enumerate(response):
            if r > 0:
                #this is a corner
                img_copy_for_corners[rowIndex, colIndex] = [255, 0, 0]
            elif r < 0:
                #this is an edge
                img_copy_for_edges[rowIndex, colIndex] = [0, 255, 0]
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    ax[0].set_title("corners found")
    ax[0].imshow(img_copy_for_corners)
    ax[1].set_title("edges found")
    ax[1].imshow(img_copy_for_edges)
    plt.show()
    
    corners = corner_peaks(CalculateWindowR.harris_response)
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation = 'nearest', cmap=plt.cm.gray)
    ax.plot(corners[:, 1], corners[:, 0], '.r', markersize=3)
    
    return img_copy_for_corners, img_copy_for_edges


def FoundCorner(window):
    rValue = CalculateWindowR(window)
    
    if(rValue > threshold):
        return True
    
    return False

crossSize = 7
crossThickness = 1
def MarkCorner(_img, x, y):
    # draw on img,  start position    end position       color     line thickness
    cv2.line(_img, (x-crossSize, y), (x+crossSize, y), (255, 0, 0), crossThickness)  # horizontal
    cv2.line(_img, (x, y-crossSize), (x, y+crossSize), (255, 255, 0), crossThickness) # vertical
    cv2.line(_img, (x, y), (x, y), (0, 0, 255), crossThickness+2) # mid point

def searchForCorner(x, y):
    if(FoundCorner(window)):
        MarkCorner(x, y)



#Displaying original image
cv2.imshow("Original image", img)

#Displaying grayscale image 
#imgGrey = ConvertToGreyscale(img, imgWidth, imgHeight, imgChannels)
#cv2.imshow("Greyscale", imgGrey)

#imgGauss = GaussFilter(imgGrey)
#cv2.imshow('Gaussian', imgGauss)

#Show image with SobelOperator applied
#imgSobel = SobelOperator(imgGrey)
#cv2.imshow('sobel',imgSobel)

#CalculateWindowR(imgSobel)
makeWindow(150,200, img)
cv2.imshow('window',window)

#Harris(imgSobel)
#cv2.imshow('Harris',imgSobel)

cv2.waitKey(0)
cv2.destroyAllWindows()