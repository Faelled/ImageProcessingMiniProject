import cv2 as cv
import numpy as np
from os import system

system('cls') # clear console

img = cv.imread("aau-city-2.JPG", 1)
dims = img.shape
imgWidth = dims[0] # amount of columns
imgHeight = dims[1] # amount of rows
imgChannels = dims[2] # amount of channels; 3 = RGB, 1 = greyscale

windowSize = 50 # size of window in pixels that we move around
step = windowSize // 2 # half (rounded down, just in case) 
window = np.zeros(shape=[windowSize, windowSize, 1], dtype=np.uint8) # make blank image to fill out as window

###### Functions have to be made before calling them (appear before use) ######

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

def FindEdges(_img):
    # maybe adjust contrast using histogram normalizing
    return cv.Canny(_img, 50, 200)

def CreateWindow(_img, x, y):
    for i in range(windowSize):
        for j in range(windowSize):
            window[i][j] = _img[i+x][j+y][0] # grab one of the values (they are all the same anyway (intensity)) to make it 1D instead of 3D, so we can calculate determinant of the window

def CalculateIntensityOfWindow(_img, x, y):
    print("nothing yet")

# Size of Cross
crossSize = 7
crossThickness = 1
def MarkCorner(_img, x, y):
    # draw on img,  start position    end position       color     line thickness
    cv.line(_img, (x-crossSize, y), (x+crossSize, y), (255, 0, 0), crossThickness)  # horizontal
    cv.line(_img, (x, y-crossSize), (x, y+crossSize), (255, 255, 0), crossThickness) # vertical
    cv.line(_img, (x, y), (x, y), (0, 0, 255), crossThickness+2) # mid point

def LookForAndDrawCorners(_imgIn, _imgOut, x, y):
    # some sweet algorithm that runs on _imgIn and then calls MarkCorner(_imgOut, x, y)
    MarkCorner(_imgOut, 50, 50)
    MarkCorner(_imgOut, 150, 150)
    MarkCorner(_imgOut, 250, 250)
    return _imgOut

def FindDeterminant():
    print("lol")
    # maybe not

CreateWindow(img, 130, 130)
imgWindow = window
cv.imshow("Window", imgWindow)

imgGrey = ConvertToGreyscale(img, imgWidth, imgHeight, imgChannels)
imgEdges = FindEdges(img)
imgCorners = LookForAndDrawCorners(imgEdges, imgGrey, imgWidth, imgHeight)

cv.imshow("Original", img)
cv.imshow("Edges Found", imgEdges)
cv.imshow("Corners Found", imgCorners)

cv.waitKey(0)