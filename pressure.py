import cv2
import numpy as np
import matplotlib.pyplot as plt


def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image, d, 50, 50)
    return image



def pressure(image):
    
    global PEN_PRESSURE

    # it's extremely necessary to convert to grayscale first
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # inverting the image pixel by pixel individually. This costs the maximum time and processing in the entire process!
    h, w = image.shape[:]
    inverted = image
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - image[x][y]

    # cv2.imshow('inverted', inverted)

    # bilateral filtering
    filtered = bilateralFilter(inverted, 3)

    # binary thresholding. Here we use 'threshold to zero' which is crucial for what we want.
    # If src(x,y) is lower than threshold=100, the new pixel value will be set to 0, else it will be left untouched!
    ret, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)
    # cv2.imshow('thresh', thresh)

    # add up all the non-zero pixel values in the image and divide by the number of them to find the average pixel value in the whole image
    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if (thresh[x][y] > 0):
                total_intensity += thresh[x][y]
                pixel_count += 1

    average_intensity = float(total_intensity) / pixel_count
    PEN_PRESSURE = average_intensity
    #print total_intensity
    # print pixel_count
    # print ("Average pen pressure: "+str(average_intensity))

    return PEN_PRESSURE

def pressure_extract(file_name):
    image = cv2.imread(file_name)
    total = pressure(image)
	
    return [total]

def main():
    image = cv2.imread('dataset/m06-106-s01-01.png')
    
    pressure(image)

    #print(zoning(image))

    cv2.waitKey(0)
    return
