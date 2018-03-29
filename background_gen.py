import numpy as np
import cv2
import sys
import time
import math
import os
import types
import argparse

# blurs a frame and crops based on boundaries
def blur_and_mask(frame, lower_bound, upper_bound, left_bound, right_bound, vidHeight, vidWidth):
    # blur image to make color uniform
    blurred = cv2.blur(frame,(9,9))

    # initialize mask to all 0s
    mask = np.zeros((vidHeight, vidWidth, 3),np.uint8)

    # use rectangle bounds for masking
    mask[lower_bound:upper_bound,left_bound:right_bound] = blurred[lower_bound:upper_bound,left_bound:right_bound]

    return mask

# computes an average frame of a video (background image)
def get_background_image(vid, numFrames, length, NUM_FRAMES_TO_SKIP, NUM_FRAMES_TO_TRIM):

    print '#' * 45
    print 'Getting background image...'

    frames2skip = NUM_FRAMES_TO_SKIP*1.0
    frames2trim = NUM_FRAMES_TO_TRIM*1.0

    # TODO: remove this (or not), but for now skipping some frames at beginning
    #       of numerosity where the feeder is moving
    j = 0
    while j < frames2skip:
        vid.read()
        j+=1
    orig_len = length
    length = length - frames2skip - frames2trim

    # set a counter
    i = 0
    #vid.set(1, 200) # TODO: what is this doing?
    _,frame = vid.read()
    frameCnt = j

    # initialize an empty array the same size of the pic to update
    update = np.float32(frame)

    #skip = int(math.floor(length/numFrames))
    skip = 0
    print 'Number of frames to skip between reads: ' + str(skip)

    # loop through every skip frames to build up average background
    while i < numFrames:

        # grab a frame
        _,frame = vid.read()
        frameCnt += 1

        # skip some frames
        #if i < numFrames-skip-1:
        #    for j in range(1,skip):
        #        vid.read()
        #        frameCnt += 1

        # main function
        #cv2.accumulateWeighted(frame,update,0.001)
        #cv2.accumulate(frame,update)
        update+=frame
        #final = cv2.convertScaleAbs(update)

        # increment the counter
        i += 1

        # print something every 100 frames so the user knows the gears are grinding
        if i%100 == 0:
            print 'Detecting background -- on frame ' + str(frameCnt) + ' of ' + str(orig_len)

    print 'Background detection complete!'
    print '#' * 45

    #final = update
    #final[:] = [x / numFrames for x in update]

    final = update/numFrames
    #final = cv2.convertScaleAbs(final)
    #final = cv2.blur(final,(9,9))

    return final

if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputvideo', help = 'Path to video file including the filename', required=True)
    ap.add_argument('-o', '--outputimage', help = 'Path to image file including the filename (should be .jpg)', required=True)
    args = vars(ap.parse_args())

    path = args['inputvideo']
    outfile = args['outputimage']

    NUM_FRAMES_TO_SKIP = 0
    NUM_FRAMES_TO_TRIM = 0
    CROP_X1 = 0
    CROP_X2 = 1296
    CROP_Y1 = 0
    CROP_Y2 = 972

    upper_bound, left_bound, right_bound, lower_bound = CROP_Y2, CROP_X1, CROP_X2, CROP_Y1

    # open the video
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print 'ERROR> Could not open :', path
        print 'Exiting...'
        sys.exit(1)

    # get some info about the video
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if length < 0:
        # manually count how many frames, this will take a little time...
        print 'Counting frames..'
        length = 0
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print 'ERROR> Could not open :', path
            print 'Exiting...'
            sys.exit(1)
        while(cap.isOpened()):
            ret,frame = cap.read()
            if ret == False:
                print 'ERROR> Did not read frame from video file'
                break
            length += 1
    print 'Number of frames: ' +  str(length)
    vidWidth  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    print 'Width: ' +  str(vidWidth)
    vidHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    print 'Height: ' +  str(vidHeight)
    fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print 'FPS: ' +  str(fps)

    # TODO: Maybe make this an input parameter
    NUM_FRAMES_FOR_BACKGROUND = length

    # error checking
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print 'ERROR> Could not open :', path
        print 'Exiting...'
        sys.exit(1)

    if NUM_FRAMES_FOR_BACKGROUND > length:
        print 'ERROR> NUM_FRAMES_FOR_BACKGROUND (' + str(NUM_FRAMES_FOR_BACKGROUND) + ') > length (' + str(length) + ')'
        print 'Changing NUM_FRAMES_FOR_BACKGROUND to equal length'
        NUM_FRAMES_FOR_BACKGROUND = length

    if NUM_FRAMES_TO_TRIM > NUM_FRAMES_FOR_BACKGROUND:
        print 'ERROR> NUM_FRAMES_TO_TRIM (' + str(NUM_FRAMES_TO_TRIM) + ') > NUM_FRAMES_FOR_BACKGROUND (' + str(NUM_FRAMES_FOR_BACKGROUND) + ')'
        print 'Exiting...'
        sys.exit(1)

    if NUM_FRAMES_TO_SKIP > NUM_FRAMES_FOR_BACKGROUND:
        print 'ERROR> NUM_FRAMES_TO_SKIP (' + str(NUM_FRAMES_TO_SKIP) + ') > NUM_FRAMES_FOR_BACKGROUND (' + str(NUM_FRAMES_FOR_BACKGROUND) + ')'
        print 'Exiting...'
        sys.exit(1)

    if (NUM_FRAMES_TO_SKIP+NUM_FRAMES_TO_TRIM) >= NUM_FRAMES_FOR_BACKGROUND:
        print 'ERROR> NUM_FRAMES_TO_SKIP+NUM_FRAMES_TO_TRIM (' + str(NUM_FRAMES_TO_SKIP+NUM_FRAMES_TO_TRIM) + ') >= NUM_FRAMES_FOR_BACKGROUND (' + str(NUM_FRAMES_FOR_BACKGROUND) + ')'
        print 'Exiting...'
        sys.exit(1)

    # TODO: Add check of max number of frames that can be used (when will it roll over)

    # calculate background image
    background = get_background_image(cap,NUM_FRAMES_FOR_BACKGROUND,length, NUM_FRAMES_TO_SKIP, NUM_FRAMES_TO_TRIM)
    #background = get_background_image(cap,(length-NUM_FRAMES_TO_TRIM),length)

    # blur and crop background
    bm_initial = blur_and_mask(background, lower_bound, upper_bound, left_bound, right_bound, vidHeight, vidWidth)

    # write background image file
    cv2.imwrite(outfile, background)
