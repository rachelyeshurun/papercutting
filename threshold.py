''' Convert a colour image to a black and white cartoon like representation of the image.
The pipeline is based on the algorithm from the paper:
Toonify: Cartoon Photo Effect Application by Kevin Dade
And also from: Stack Overflow: https://stackoverflow.com/questions/1357403/how-to-cartoon-ify-an-image-programmatically
'''
import errno
import os
import sys

import numpy as np
import cv2

from utils import show_image


import numpy as np
import cv2


#!/usr/bin/env python

def gray2binary(image, canny_lower = 100, canny_upper = 200, sigma_color = 70, sigma_space = 11, thresholds = None):
    ''' main function to threshold in a special way '''

    # Follow 'toonify' algorithm, but quantize to only 2 colours: black and white
    # Create two 'layers' and then combine them.
    # Layer 1 is the edges: median filter with 7x7 kernel, then Canny edge detection, then dilate with small 2x2 structuring element,
    # then filter out 'little' edges (reduce clutter)
    # Layer 2 is the colours: Bilateral filter, then median filter with 7x7 kernel, then quantize to 2 colours: black and white


    # show_image(layer1, 'luminance')

    # layer1 = dither_floyd_steinberg(layer1)
    # show_image(layer1, 'Results of floyd-steinberg')

    # Median filter with 7x7 kernel
    layer1 = median_filter_7_by_7(image)
    # show_image(layer1, 'median')

    # Canny edge detection - to get the 1-pixel wide edge lines
    layer1 = canny_edge_detection(layer1, canny_lower, canny_upper)
    #show_image(layer1, 'Canny')

    layer1 = dilate_with_2_by_2(layer1)
    #show_image(layer1, 'Dilated')

    # skipping the 'edge filter' step, Canny hysteresis params do this well enough
    # layer1 = filter_edges(layer1)
    # show_image(layer1, 'mask')

    # Bilateral filter on luminance input imaage (instead of on colour like in toonify paper)
    layer2 = bilateral_filter(image, sigma_color, sigma_space)
    # show_image(layer2, 'Results of bilateral filtering (blur but preserve edges)')

    layer2 = median_filter_7_by_7(layer2)
    # show_image(layer2, 'Results of 7x7 median filter')

    layer2 = quantize_to_binary(layer2, thresholds)
    # show_image(layer2, 'Results of quantizing')
    
    # ..and now.. combine!
    # layer1 white lines should add black lines in white places, but not white lines to black.. so invert the canny result and bitwise AND with the blurred and thresholded image
    return cv2.bitwise_and(np.bitwise_not(layer1), layer2)



def median_filter_7_by_7(image):
    return cv2.medianBlur(image, 7)

# based on this idea: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def canny_edge_detection(image, lower = 100, upper = 200):
    # minval is low in order to get more lines that are connected to sure lines
    # maxval is also relatively low in order to catch the fine lines of faces that give it character
    # median = np.median(image)
    # sigma = 0.33
    # # print('median, sigma', median, sigma)
    # lower = int(max(0, (1.0 - sigma) * median))
    # upper = int(min(255, (1.0 + sigma) * median))
    # print('median, lower, upper', median, lower, upper)
    return cv2.Canny(image, lower, upper, edges=None, apertureSize=3, L2gradient=True)

def dilate_with_2_by_2(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def filter_edges(image, area_threshold = 50):
    '''
    inspired by https://stackoverflow.com/questions/47055771/how-to-extract-the-largest-connected-component-using-opencv-and-python
    '''
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    mask = np.zeros_like(labels)
    return mask



def bilateral_filter(image, color = 70, space = 11):
    # todo: play with parameters
    # the 1st parameter is size of gaussian kernel (num pixels to blur) - making negative so that it's computed by opencv acc. to sigmaSpace
    # the 2nd parameter is how many intensity values is considered different enough that it's not included in the blurring
    # the 3rd parameter is size of neighborhood to blur around - but only blur the pixels with similar enough values acc. to sigmaColor
    return cv2.bilateralFilter(image, -1, sigmaColor=color, sigmaSpace=space)


def quantize_to_binary(image, thresholds = None):

    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(image)

    if thresholds is None:
        median = np.median(gray)
        gray[np.where(gray < median)] = 0
        gray[np.where(gray >= median)] = 255
        return gray

    thresholded_image = np.zeros_like(gray)

    # thresholding from outside the face to inside, add black
    for thresh in thresholds:
        mask, thresh_val = thresh
        # print('threshold val: ', thresh_val)
        # show_image(mask, 'mask')
        masked = cv2.bitwise_and(gray, mask)
        # show_image(masked, 'masked')
        masked[np.where(masked < thresh_val)] = 0
        masked[np.where(masked >= thresh_val)] = 255
        # show_image(masked, 'masked')
        thresholded_image = cv2.bitwise_or(thresholded_image, masked)
        # show_image(thresholded_image, 'thresholded_image')


    return thresholded_image

        # show_image(gray, 'after thresholding')
    # # print ('threshold for black is: ', threshold)
    # to_black = np.where(gray < threshold)
    # gray[to_black] = 0
    # to_white = np.where(gray >= threshold)
    # gray[to_white] = 255
    return gray

def dither_floyd_steinberg(image):
    ''' input any image, output is dithered with the error diffused '''

    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(image)

    rows, cols = gray.shape
    """ Floyd-Steinberg Dithering algorithm, see:
        http://en.wikipedia.org/wiki/Floyd-Steinberg
        Code inspired by the 'Dither' class implementation: http://code.activestate.com/recipes/576788-floyd-steinberg-dithering-algorithm/
    """
    # Inspired by the 'Dither' class implementation: http://code.activestate.com/recipes/576788-floyd-steinberg-dithering-algorithm/
    for row in range(rows - 1):
        for col in range(cols - 1):
            oldpixel = gray[row, col]
            if oldpixel < 127:
                newpixel = 0
            else:
                newpixel = 255
                gray[row, col] = newpixel
            quant_error = oldpixel - newpixel
            if col < cols:
                gray[row, col + 1] += quant_error * 7 / 16
            if row < rows and col > 0:
                gray[row + 1, col - 1] += quant_error * 3 / 16
            if row < rows:
                gray[row + 1, col] += quant_error * 5 / 16
            if row < rows and col < cols:
                gray[row + 1, col + 1] += quant_error * 1 / 16
    return gray
