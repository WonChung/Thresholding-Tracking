########################################################################
#
# File:   regions.py
# Author: Matt Zucker
# Date:   January, 2012 (Updated January, 2017)
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This file shows how to do connected component analysis with OpenCV
# and cvk2.

# Do Python 3-style printing
from __future__ import print_function

import cv2
import numpy
import sys
import cvk2

# Get an image from the command line and load it.
if len(sys.argv) < 2:
    print('supply an image filename (e.g. screws_thresholded.png)')
    sys.exit(1)

image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

# We will work on a copy of the image, because findContours below
# modifies the image destructively.
work = image.copy()

# Create an RGB display image which to show the different regions.
display = numpy.zeros((image.shape[0], image.shape[1], 3),
                      dtype='uint8')

# Get the list of contours in the image. See OpenCV docs for
# information about the arguments.
image, contours, hierarchy = cv2.findContours(work, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)

print('found', len(contours), 'contours')

# The getccolors function from cvk2 supplies a useful list
# of different colors to color things in with.
ccolors = cvk2.getccolors()

# Define the color white (used below).
white = (255,255,255)

# For each contour in the image
for j in range(len(contours)):

    # Draw the contour as a colored region on the display image.
    cv2.drawContours( display, contours, j, ccolors[j % len(ccolors)], -1 )

    # Compute some statistics about this contour.
    info = cvk2.getcontourinfo(contours[j])

    # Mean location and basis vectors can be useful.
    mu = info['mean']
    b1 = info['b1']
    b2 = info['b2']

    # Annotate the display image with mean and basis vectors.
    cv2.circle( display, cvk2.array2cv_int(mu), 3, white, 1, cv2.LINE_AA )
    
    cv2.line( display, cvk2.array2cv_int(mu), cvk2.array2cv_int(mu+2*b1),
              white, 1, cv2.LINE_AA )
    
    cv2.line( display, cvk2.array2cv_int(mu), cvk2.array2cv_int(mu+2*b2),
              white, 1, cv2.LINE_AA )

# Display the output image and wait for a keypress.
cv2.imshow('Regions', display)
while cv2.waitKey(5) < 0: pass

