# -*- coding: utf-8 -*-
import numpy as np
import cv2

imgInput = cv2.imread('aau-city-2.jpg',1)

#Variables
dims = imgInput.shape #Dimensions of image?
imgWidth = dims[0] # amount of columns
imgHeight = dims[1] # amount of rows
imgChannels = dims[2] # amount of channels; 3 = RGB, 1 = greyscale

windowSize = 3 # size of window in pixels that we move around
step = windowSize // 2 # half (rounded down, just in case) 

hej = 0

#Converting to grayscale - defining grayscale func.
def ConvertToGreyscale(_img, x, y, z):
    _grey = _img.copy()
    _testgrey = np.zeros(shape=[imgWidth, imgHeight, 1], dtype=np.uint8)
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
    kernelSize = 5 
    offset = kernelSize//2 
    #gaussKernel= np.array([[1,2,1],
    #              [2,4,2],
    #              [1,2,1]])
    
    gaussKernel = np.array([[1,4,7,4,1],
                  [4,16,26,16,4],
                  [7,26,41,26,7],
                  [4,16,26,16,4],
                  [1,4,7,4,1]])
       
       
    for j in range (offset, y-offset): #Loop through every pixel in the image, the offset is used to avoid the edges of the image
        for i in range (offset, x-offset):
            newValue = 0.0
            
            for l in range(0, kernelSize):#Loop through the kernels rows and columns
                for k in range(0, kernelSize):
                    newValue += img[i+k-offset][j+l -offset][0]*gaussKernel[l][k] 
  
            newValue=newValue//(273)#16 is sum of the kernel
            img[i][j] = newValue #Set the pixel to the new value calculated for it
            
    return img;


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
    window = np.zeros(shape=[windowSize, windowSize, 1], dtype=np.uint8) # make blank image to fill out as window
    row = 0
    col = 0
    
    for i in range (x-step, x+step):
        col = 0
        for j in range (y-step, y+step):
            window[row][col] = img[i][j][0]
            col +=1
        row +=1
    return window
                      

#def DisplaceWindow(x,y, img):
#    for i in range ()


def CalculateWindowR(window):
    k = 0.04
    
    Ix = SobelGradientX(window, 1, 1)
    Iy = SobelGradientY(window, 1, 1)
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    detA = Ixx * Iyy - Ixy * Ixy
    traceA = Ixx + Iyy
    
    harris_response = detA - k * (traceA * traceA)
    
    return harris_response
    #return det - k * (trace * trace) #harris_response


def Harris(img):    

    for x in range (10, imgWidth - 10):
            for y in range (10, imgHeight - 10):
                window = makeWindow(x, y, img)
                SearchForCorner(x, y, window)
    
    # Gå gennem billed hver halve billedlængde
    # Ved hvert step, lav et vindue, og displace 4 vinduer i hvert diagonal retning; stop når R-værdien for et vindue er høj nok (??) eller når man har lavet 4 vinduer
    # Hvis R-værdien er høj nok, marker som hjørne, ellers fortsæt og lav et billed af næste step

    return img #???



def FoundCorner(window): 
    rValue = CalculateWindowR(window)
     
    
    if(rValue > 40000000): #dont even question it
        return True
    
    return False


crossSize = 3
crossThickness = 1
def MarkCorner(_img, x, y):
    # draw on img,  start position    end position       color     line thickness
    cv2.line(_img, (x-crossSize, y), (x+crossSize, y), (255, 0, 0), crossThickness)  # horizontal
    cv2.line(_img, (x, y-crossSize), (x, y+crossSize), (255, 255, 0), crossThickness) # vertical
    cv2.line(_img, (x, y), (x, y), (0, 0, 255), crossThickness+2) # mid point


def SearchForCorner(x, y, window):
    if(FoundCorner(window)):
        MarkCorner(imgInput, y, x)
        
def MakeBinary(img, y, x):

    for i in range (0, x):
        for j in range (0, y):
            if (img[i][j][0] < 125):
                img[i][j] = 0
            else:
                img[i][j] = 255
            
    return img
    


#Displaying grayscale image 
imgGrey = ConvertToGreyscale(imgInput, imgWidth, imgHeight, imgChannels)
#cv2.imshow("Greyscale", imgGrey)

imgGauss = GaussFilter(imgGrey)
cv2.imshow('BLUR SONG #2',imgGauss)

#Show image with SobelOperator applied
imgSobel = SobelOperator(imgGauss)
binaryImg = MakeBinary(imgSobel, imgHeight, imgWidth)

imgHarris = Harris(binaryImg)
cv2.imshow('Harris',imgHarris)

cv2.imshow("Original image", imgInput)

cv2.waitKey(0)
cv2.destroyAllWindows()