########################################################################
#
# File:   iSpot.py
# Author: William Lee (and Won Chung) (w/ code taken from Matt Zucker's capture.py)
# Date:   February, 2017
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This program demonstrates how to use the VideoCapture and
# VideoWriter objects from OpenCV.
#
# Usage: the program can be run with a filename or a single integer as
# a command line argument.  Integers are camera device ID's (usually
# starting at 0).  If no argument is given, tries to capture from
# the default input 'bunny.mp4'

# Do Python 3-style printing
from __future__ import print_function

import cv2
import numpy as np
import sys
import struct
import pdb
import cvk2

#Displays a message on screen, warning the user to spot the lifter
def warnToSpot():
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display,'Warning: Lifter needs a spot!',(75,75), font, 1,(255,255,255),2)

# Figure out what input we should load:
input_device = None
if len(sys.argv) > 1:
    input_filename = sys.argv[1]
    try:
        input_device = int(input_filename)
    except:
        pass
else:
    print('Using default input. Specify a device number to try using your camera, e.g.:')
    print()
    print('  python', sys.argv[0], '0')
    print()
    input_filename = 'bunny.mp4'
# Choose camera or file, depending upon whether device was set:
if input_device is not None:
    capture = cv2.VideoCapture(input_device)
    if capture:
        print('Opened camera device number', input_device, '- press Esc to stop capturing.')
else:
    capture = cv2.VideoCapture(input_filename)
    if capture:
        print('Opened file', input_filename)

# Bail if error.
if not capture or not capture.isOpened():
    print('Error opening video capture!')
    sys.exit(1)
# Fetch the first frame and bail if none.
ok, frame = capture.read()
if not ok or frame is None:
    print('No frames in video')
    sys.exit(1)
#Grab width/height
w = frame.shape[1]
h = frame.shape[0]

# Now set up a VideoWriter to output video. (dependent on w, h above)
fps = 30 #do we want to downgrade?
# One of these combinations should hopefully work on your platform:
#fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi')
fourcc, ext = (cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 'mov')
filename = 'captured.'+ext
writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
if not writer:
    print('Error opening writer')
else:
    print('Opened', filename, 'for output.')
    writer.write(frame)
# Loop until movie is ended or user hits ESC:

################################################################################################
#                                    Computations Below                                        #
################################################################################################

#The bar centroid is defined as the midpoint of the line connecting the centroids of both hands (drawn with white dots in the video)
#We use the bar centroids to approximate the speed of the bar and determine if the lifter needs a spot (when the velocity is close to 0)
barCentroids = [] #Keep track of the locations of previous bar centroids. Used to calculate bar velocity.
velocity = [] #Stores the velocities of the bar centroids. (not actual velocity, but a relatively close measure: distance between position at time x and position at time y (not normalized for time))
frameNumber=0; #Used to keep track of the frame number of the video
while 1:
    frameNumber = frameNumber + 1 #increment frameNumber
    #print('frame number ', frameNumber) #print current frame number if desired
    # Get the frame.
    ok, frame = capture.read(frame)
    # Bail if none.
    if frame is None:
        print('Video Finished!')
        break
    if not ok:
        print('Bad frame in video! Aborting!')
        break

    #RGB thresholding to isolate the hands in the frame
    video = frame
    #convert to HSV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range of colors to Get Only Gloves without Weights. Other thresholding options commented below.
    lower_blue = np.array([85,70,20])
    upper_blue = np.array([100,255,255])

    # To Get Only Gloves
    # lower_blue = np.array([85,70,20])
    # upper_blue = np.array([130,255,255])

    # To Get Dumbbells and Gloves
    # lower_blue = np.array([40,80,20])
    # upper_blue = np.array([130,255,255])

    # Get Arms and Gloves
    # lower_blue = np.array([0,90,140])
    # upper_blue = np.array([130,255,255])

    # Get Arm Only Or Skin Only
    # lower_blue = np.array([0,90,110])
    # upper_blue = np.array([50,255,255])

    #RGB threshold using a mask and our color ranges
    mask = cv2.inRange(frame, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('mask',mask)

    #Here, we use morphological Operators to remove noise
    kernel = np.ones((10,10), np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    # cv2.imshow('Erosion',erosion)
    # cv2.imshow('Dilation',dilation)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #this is the morphological operator we used
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Opening',opening)
    # cv2.imshow('Closing',closing)


    #Here, we do connected Components Analysis
    #Make a copy of the frame cleaned by morphological operators to perform connected components analysis on
    bimodalImg = opening.copy() #we chose to use the output from the opening morphological operator
    #initalize display w/ zeros
    display = np.zeros((bimodalImg.shape[0], bimodalImg.shape[1], 3),
                      dtype='uint8')

    # Get the list of contours in the image. See OpenCV docs for
    # information about the arguments. (taken from example code)
    bimodalImg, contours, hierarchy = cv2.findContours(bimodalImg, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)
    #print contours for each frame if wanted. 
    #print('found', len(contours), 'contours')

    # The getccolors function from cvk2 supplies a useful list
    # of different colors to color things in with.
    ccolors = cvk2.getccolors()

    #define colors for the trail left behind by the centroid (we paint the location of the previous)
    #10 centroids, with older locations being a darker shade of white to have a "trail" effect
    trail = [(255, 255, 255),
    (240, 240, 240),
    (225, 225, 225),
    (210, 210, 210),
    (195, 195, 195),
    (180, 180, 180),
    (165, 165, 165),
    (150, 150, 150),
    (135, 135, 135),
    (120, 120, 120),
    (105, 105, 105)]

    # Define the color white, pink, red (used below).
    white = (255,255,255)
    pink = (247,116,182)

    #We expect to have 2 contours in our image (1 for each hand). Store their centroids in this array
    #so we can later access their centroids and connect them with a line. 
    centroids = []
    # For each contour in the image
    for j in range(len(contours)):
        # Draw the contour as a colored region on the display image.
        cv2.drawContours(display, contours, j, ccolors[j % len(ccolors)], -1 )
        # Compute some statistics about this contour.
        info = cvk2.getcontourinfo(contours[j])
        # Mean location and basis vectors can be useful.
        mu = info['mean']
        b1 = info['b1']
        b2 = info['b2']
        centroids.append(mu) #Add the centroid coordinate to our centroids array

        # Annotate the display image with mean and basis vectors.
        cv2.circle( display, cvk2.array2cv_int(mu), 3, white, 1, cv2.LINE_AA )
        cv2.line( display, cvk2.array2cv_int(mu), cvk2.array2cv_int(mu+2*b1),
                  white, 1, cv2.LINE_AA )
        cv2.line( display, cvk2.array2cv_int(mu), cvk2.array2cv_int(mu+2*b2),
                  white, 1, cv2.LINE_AA )
    #draw line between centroids
    cv2.line(display, cvk2.array2cv_int(centroids[0]), cvk2.array2cv_int(centroids[1]), pink, 1, cv2.LINE_AA)

    #calculate the midpoint between the centroids of the hands (barCentroid)
    midpoint = cvk2.array2cv_int(0.5*(centroids[0]+centroids[1]))

    #Use barCentroids to keep track of previous bar centroids to calculate bar velocities
    barCentroids.append(midpoint)

    #Draw the barCentroid trail
    for i in range(0,len(barCentroids)):
        #if less than 10 frames have elapsed, we set velocity to 0 and don't paint any circles
        #(The lifter will not need a spot 10 frames in, and the calculated velocity 10 frames in would
        #be very very close to 0. 
        if frameNumber <= 10:
            numCircles = 0 #the number of circles in the trail to draw
        else:
            #otherwise, we draw a trail composed of 10 circles
            numCircles = 10
        for k in range(1, numCircles):
            cv2.circle(display, barCentroids[len(centroids)-numCircles-1], 1, trail[10-k], 1, cv2.LINE_AA )
            numCircles = numCircles-1

    #Calculate Velocity of the Bar (barCentroid) in this frame    

    if frameNumber <= 10:
        velocity.append(0) #set bar velocity to 0 if less than 10 frames in 
    else:
        if frameNumber<100:
            #if less than 100 frames have elapsed, we calculate velocity as distance changed over 10 frames
            position1 = barCentroids[len(barCentroids)-1]
            position2 = barCentroids[len(barCentroids)-10]
        else:
            #if more than 100 frames have elapsed, we calculate the velocity as distance changed over 20 frames
            position1 = barCentroids[len(barCentroids)-1]
            position2 = barCentroids[len(barCentroids)-20]

        #Calculate the velocity of the barcentroid at the current frame
        position = np.subtract(position1,position2)
        velocity.append(np.sqrt((position[0]*position[0]+position[1]*position[1])))

        #Scalar value determining how aggressively to spot, a tradeoff between decapitation risk and gains
        decapitationVsGainsTradeoff=10
        if velocity[frameNumber-1] < decapitationVsGainsTradeoff:               
            warnToSpot()
    # plt.plot(barCentroids, velocity)
    # plt.title('Tracjectories')
    # plt.grid(True)
    # plt.show()
    cv2.imshow('Regions', display)

    ################################################################################################
    #                          Write the newly modified frame to the writer                        #
    ################################################################################################
    # Write if we have a writer.
    if writer:
        writer.write(frame)
    # Throw it up on the screen.
    cv2.imshow('Video', video)
    # cv2.imshow('average', average.astype(np.uint8))
    # cv2.imshow('diff matrix', diffMatrix.astype(np.uint8))
    # cv2.imshow('Gray Average', grayAverage.astype(np.uint8))
    # cv2.imshow('Gray Frame', grayFrame.astype(np.uint8))
    # cv2.imshow('Temporal Threshold', temporalThreshold.astype(np.uint8))
    # cv2.imshow('absdiff', diffMatrix.astype(np.uint8))

    # Delay for 5ms and get a key
    k = cv2.waitKey(5)
    # Check for ESC hit:
    if k % 0x100 == 27:
        break