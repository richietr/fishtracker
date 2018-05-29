import numpy as np
import cv2
import sys
import time
import math
import os
import csv
import json
import types
import argparse

def parse_json(config_file):
    if os.path.exists(config_file):
        with open(os.path.join(config_file), 'r') as jsonFile:
            jsonString = jsonFile.read()
            paramsDict = json.loads(jsonString)
            for k,v in paramsDict.iteritems():
                if type(v) == types.UnicodeType:
                    path_elems = v.split('\\')
                    newPath = os.path.join(*path_elems)
                    paramsDict[k] = newPath
        config_dict = paramsDict

        # check error conditions
        if config_dict is None:
            print 'Error> Config dictionary is None, problem parsing config file (json)'

        return config_dict
    else:
        print 'Error> No config file (json), will use hard-coded defaults'
        return None

def true_distance(a, b):
    d = math.sqrt((float(a[0]) - b[0])**2 + (float(a[1]) - b[1])**2)
    return d

# crops based on boundaries
def apply_mask(frame, lower_bound, upper_bound, left_bound, right_bound, vidHeight, vidWidth):

    # initialize mask to all 0s
    mask = np.zeros((vidHeight, vidWidth, 3),np.uint8)

    # use rectangle bounds for masking
    mask[lower_bound:upper_bound,left_bound:right_bound] = frame[lower_bound:upper_bound,left_bound:right_bound]

    return mask

def apply_special_mask(frame, corner_x1, corner_x2, corner_y1, corner_y2, target_x1, target_x2, corner_y3, corner_y4, vidHeight, vidWidth):

    # initialize mask to all 0s
    mask = np.zeros((vidHeight, vidWidth, 3),np.uint8)

    # use rectangle bounds for masking
    mask[corner_y1:corner_y2,corner_x1:corner_x2] = frame[corner_y1:corner_y2,corner_x1:corner_x2]
    mask[corner_y2:corner_y3,target_x1:target_x2] = frame[corner_y2:corner_y3,target_x1:target_x2]
    mask[corner_y3:corner_y4,corner_x1:corner_x2] = frame[corner_y3:corner_y4,corner_x1:corner_x2]

    return mask

def apply_test_mask(frame, mirror_x1, mirror_x2, upper_mirror_y2, lower_mirror_y1, left_screen_x2, right_screen_x1, screen_y1, screen_y2, vidHeight, vidWidth):

    # initialize mask to all 0s
    #mask = np.zeros((vidHeight, vidWidth, 3),np.uint8)

    zeros = np.zeros((vidHeight, vidWidth, 3),np.uint8)

    # use rectangle bounds for masking
    # upper mirror
    frame[0:upper_mirror_y2,mirror_x1:mirror_x2] = zeros[0:upper_mirror_y2,mirror_x1:mirror_x2]
    # lower mirror
    frame[lower_mirror_y1:vidHeight-1,mirror_x1:mirror_x2] = zeros[lower_mirror_y1:vidHeight-1,mirror_x1:mirror_x2]
    # left screen
    frame[screen_y1:screen_y2,0:left_screen_x2] = zeros[screen_y1:screen_y2,0:left_screen_x2]
    # right screen
    frame[screen_y1:screen_y2,right_screen_x1:vidWidth-1] = zeros[screen_y1:screen_y2,right_screen_x1:vidWidth-1]

    #edges
    frame[0:20,0:vidWidth-1] = zeros[0:20,0:vidWidth-1]
    frame[0:vidHeight-1,0:20] = zeros[0:vidHeight-1,0:20]
    frame[vidHeight-21:vidHeight-1,0:vidWidth-1] = zeros[vidHeight-21:vidHeight-1,0:vidWidth-1]
    frame[0:vidHeight-1,vidWidth-21:vidWidth-1] = zeros[0:vidHeight-1,vidWidth-21:vidWidth-1]

    return frame

# blurs a frame and crops based on boundaries
def blur_and_mask(frame, lower_bound, upper_bound, left_bound, right_bound, vidHeight, vidWidth):
    # blur image to make color uniform
    blurred = cv2.blur(frame,(9,9))

    # initialize mask to all 0s
    mask = np.zeros((vidHeight, vidWidth, 3),np.uint8)

    # use rectangle bounds for masking
    mask[lower_bound:upper_bound,left_bound:right_bound] = blurred[lower_bound:upper_bound,left_bound:right_bound]

    return mask

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 25:
                return True
            elif i==row1-1 and j==row2-1:
                return False

def merge_contours(frame, thresh):
    contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
    #image, contours, hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    if LENGTH > 1:

        for i,cnt1 in enumerate(contours):
            x = i
            if i != LENGTH-1:
                for j,cnt2 in enumerate(contours[i+1:]):
                    x = x+1
                    dist = find_if_close(cnt1,cnt2)
                    if dist == True:
                        val = min(status[i],status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x]==status[i]:
                            status[x] = i+1

        unified = []
        maximum = int(status.max())+1
        for i in xrange(maximum):
            pos = np.where(status==i)[0]
            if pos.size != 0:
                cont = np.vstack(contours[i] for i in pos)
                hull = cv2.convexHull(cont)
                unified.append(hull)

        cv2.drawContours(frame,unified,-1,(128,0,128),2)
        cv2.drawContours(thresh,unified,-1,255,-1)

        return unified


# returns center of fish
def find_fish(frame,totalVideoPixels, unified, min_area_pixels, max_area_pixels, min_height_pixels, max_height_pixels):

    # find all contours in the frame
    contours = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #image,contours,hier = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print 'number of contours: ' + str(len(contours[0]))
    if len(contours) > 1 and unified is not None:
        contours = unified
    else:
        contours = contours[0]

    if len(contours) > 0:
        area_list = []
        for z in contours:
            try:
                # calculate area and bounding rectangle (to determine length)
                area_list.append(cv2.contourArea(z))
                x,y,w,h = cv2.boundingRect(z)
            except:
                pass

        potential_tracks = []
        for area in area_list:
            #if area < max_area_pixels*1.0 and area > min_area_pixels*1.0 and max(h,w) < max_height_pixels and max(h,w) > min_height_pixels:
            if area < max_area_pixels*1.0 and area > min_area_pixels*1.0:
                idx = area_list.index(area)
                potential_tracks.append(contours[idx])
            #else:
                #print 'Area=' + str(area) + ', MaxH=' + str(max(h,w))
                #print 'Area=' + str(area)


        #print 'Number of potential tracks: ' + str(len(potential_tracks)) + ' of ' + str(len(area_list))

        if len(potential_tracks) > 1 or len(potential_tracks) == 0:
            return None
        else:

            #print 'Area=' + str(area) + ', MaxH=' + str(max(h,w))

            largestCon = area_list.index(max(area_list))

            m = cv2.moments(contours[largestCon])
            if int(m['m00']) == 0:
                return None
            else:
                centroid_x = int(m['m10']/m['m00'])
                centroid_y = int(m['m01']/m['m00'])
                return((centroid_x,centroid_y))

    else:
        return None

# computes an average frame of a video (background image)
def get_background_image(vid,numFrames,length, NUM_FRAMES_TO_SKIP, NUM_FRAMES_TO_TRIM):

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

    skip = int(math.floor(length/numFrames))
    print 'Number of frames to skip between reads: ' + str(skip)

    # loop through every skip frames to build up average background
    while i < numFrames:

        # grab a frame
        _,frame = vid.read()
        frameCnt += 1

        # skip some frames
        if i < numFrames-skip-1:
            for j in range(1,skip):
                vid.read()
                frameCnt += 1

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


###############################################################################
#
#  TODO items
# - tracking multiple targets
# - handle shadows
# - code cleanup (think about how could be easily configured for sociality)
#
###############################################################################

class Tracker:

    def __init__(self, vid_file, config_file, fish_json, grid_json, show_images):

        self.vid_file = vid_file
        self.config_file = config_file
        self.show_images = show_images

        print 'Video file: ' + str(vid_file)
        print 'Config file (json): ' + str(config_file)
        print 'Fish json: ' + str(fish_json)
        print 'Grid json: ' + str(grid_json)

        if show_images:
            print 'Showing images!'
        else:
            print 'NOT showing images!'

        self.line6 = None
        self.line7 = None
        self.line9 = None
        self.line10 = None
        self.line11 = None
        self.line12 = None

        # Get fish name from video file
        path_strip = os.path.splitext(vid_file)[0]
        if os.name == 'nt': # Windows
            path_parts = path_strip.split('\\')
        else: # Linux
            path_parts = path_strip.split('/')
        filename = path_parts[len(path_parts)-1]
        filename_parts = filename.split('_')
        fish = filename_parts[4]

        self.set_grid_lines(self, fish, fish_json, grid_json)

    @staticmethod
    def set_grid_lines(self, fish, fish_json_filename, grid_json_filename):

        fish_json = parse_json(fish_json_filename)
        if fish_json is not None:
            if fish in fish_json:
                tmp_dict = fish_json[fish]
                if 'cam_node' in tmp_dict:
                    pie = tmp_dict['cam_node']
                else:
                    print 'ERROR> cam_node not specified in ' + fish_json_filename + ' for ' + fish_json
                    print 'Exiting...'
                    sys.exit(1)
            else:
                print 'ERROR> ' + fish + ' is not in ' + fish_json_filename
                print 'Exiting...'
                sys.exit(1)
        else:
            print 'ERROR> ' + fish_json_filename + ' is None'
            print 'Exiting...'
            sys.exit(1)

        grid_json = parse_json(grid_json_filename)
        if grid_json is not None:
            if pie in grid_json:
                tmp_dict = grid_json[pie]
                if "line6" in tmp_dict:
                    self.line6 = eval(tmp_dict["line6"])
                else:
                    print 'ERROR> line6 is not defined in ' + grid_json_filename + ' for '+ pie
                    print 'Exiting...'
                    sys.exit(1)
                if "line7" in tmp_dict:
                    self.line7 = eval(tmp_dict["line7"])
                else:
                    print 'ERROR> line7 is not defined in ' + grid_json_filename + ' for '+ pie
                    print 'Exiting...'
                    sys.exit(1)
                if "line9" in tmp_dict:
                    self.line9 = eval(tmp_dict["line9"])
                else:
                    print 'ERROR> line9 is not defined in ' + grid_json_filename + ' for '+ pie
                    print 'Exiting...'
                    sys.exit(1)
                if "line10" in tmp_dict:
                    self.line10 = eval(tmp_dict["line10"])
                else:
                    print 'ERROR> line10 is not defined in ' + grid_json_filename + ' for '+ pie
                    print 'Exiting...'
                    sys.exit(1)
                if "line11" in tmp_dict:
                    self.line11 = eval(tmp_dict["line11"])
                else:
                    print 'ERROR> line11 is not defined in ' + grid_json_filename + ' for '+ pie
                    print 'Exiting...'
                    sys.exit(1)
                if "line12" in tmp_dict:
                    self.line12 = eval(tmp_dict["line12"])
                else:
                    print 'ERROR> line12 is not defined in ' + grid_json_filename + ' for '+ pie
                    print 'Exiting...'
                    sys.exit(1)

            else:
                print 'ERROR> ' + pie + ' is not in ' + grid_json_filename
                print 'Exiting...'
                sys.exit(1)
        else:
            print 'ERROR> ' + grid_json_filename + ' is None'
            print 'Exiting...'
            sys.exit(1)




    def run_numerosity_tracker(self):

        #line1 =[(66,231),(1218,231)]
        #line2 =[(66,292),(1218,292)]
        #line3 =[(66,734),(1218,724)]
        #line4 =[(66,793),(1218,783)]
        #line5 =[(66,231),(66,793)]
        #line6 =[(365,292),(365,732)]
        #line7 =[(903,292),(903,728)]
        #line8 =[(1218,231),(1218,783)]
        #line9 =[(557,730),(557,789)]
        #line10 =[(709,730),(709,789)]
        #line11 =[(557,231),(557,292)]
        #line12 =[(718,231),(718,292)]

        print 'line6: ' + str(self.line6)
        print 'line7: ' + str(self.line7)
        print 'line9: ' + str(self.line9)
        print 'line10: ' + str(self.line10)
        print 'line11: ' + str(self.line11)
        print 'line12: ' + str(self.line12)



        print time.strftime('%X %x %Z')
        start_time = time.time()

        path = self.vid_file

        # Parse out filepath
        path_strip = os.path.splitext(path)[0]
        if os.name == 'nt': # Windows
            path_parts = path_strip.split('\\')
        else: # Linux
            path_parts = path_strip.split('/')
        filename = path_parts[len(path_parts)-1]
        filename_parts = filename.split('_')

        # Parse out filename
        species = filename_parts[0]
        round_num = filename_parts[1]
        std_len = filename_parts[2]
        sex = filename_parts[3]
        fishid = filename_parts[4]
        day = filename_parts[5]
        session = filename_parts[6]
        stimulus = filename_parts[7]
        that_stimulus = filename_parts[8]
        proportion = filename_parts[9]
        fedside = filename_parts[10]
        correctside = filename_parts[11]

        # defaults
        gen_cowlog = False
        gen_tracker_video = False
        DIFF_THRESHOLD = 15
        FISH_AREA_MIN_PIXELS = 200
        FISH_AREA_MAX_PIXELS = 2000
        FISH_HEIGHT_MIN_PIXELS = 10
        FISH_HEIGHT_MAX_PIXELS = 100
        SECS_B4_LOST = 1
        FEED_DELAY = 10+2 # secs
        FEED_DURATION = 220-2 # secs
        TOTAL_TIME = 245 # secs
        SCREEN_DELAY = 30
        DELAY_TRACKING_SECS = 0 # secs to delay starting tracker
        RULE1_SECS = 30 # once detection starts, seconds fish must be detected before declare video not trackable
        RULE2_SECS = 30 # if the fish is lost for the final RULE2SECS then declare video not trackable
        RULE3_SECS = 30 # if at any point if tracking is lost for RULE3_SECS then decare video not trackable
        NUM_FRAMES_FOR_BACKGROUND = 1000
        PRE_SCREEN_EDGE_BUFFER_X = 175 #pixels
        PRE_SCREEN_EDGE_BUFFER_Y = 175 #pixels
        EDGE_BUFFER_X = 75 #pixels
        EDGE_BUFFER_Y = 75 #pixels
        CROP_X1 = 0
        CROP_X2 = 1296
        CROP_Y1 = 0
        CROP_Y2 = 972
        #TANK_LENGTH_CM = 40.0 #40.64
        #TANK_WIDTH_CM = 20.0 #21.59
        TANK_UPPER_LEFT_X = 0
        TANK_UPPER_LEFT_Y = 0
        TANK_LOWER_LEFT_X = 0
        TANK_LOWER_LEFT_Y = 972
        TANK_UPPER_RIGHT_X = 1296
        TANK_UPPER_RIGHT_Y = 0
        TANK_LOWER_RIGHT_X = 1296
        TANK_LOWER_RIGHT_Y = 972
        #MIRROR_LENGTH_CM = 5.08 # 2"
        #MIRROR_ZONE_WIDTH_CM = 3
        TRACKING_WINDOW_LEN = 100 # TODO: Use cm instead of pixels
        #TARGET_ZONE_CM = 11
        HIGH_STIMULUS_LETTER = ['I']
        LOW_STIMULUS_LETTER = ['O']
        FREEZE_TIME_MIN_SECS = 3 # freeze must be 3 secs
        FREEZE_WINDOW_LEN = 40 # TODO: Use cm instead of pixels

        # Check if defaults overriden by numerosity_tracker_config_NEW.json
        config_json = parse_json(self.config_file)
        if config_json is not None:
            print 'Using params from config.json'
            if 'DIFF_THRESHOLD' in config_json:
                DIFF_THRESHOLD = int(config_json['DIFF_THRESHOLD'])
            if 'FISH_AREA_MIN_PIXELS' in config_json:
                FISH_AREA_MIN_PIXELS = int(config_json['FISH_AREA_MIN_PIXELS'])
            if 'FISH_AREA_MAX_PIXELS' in config_json:
                FISH_AREA_MAX_PIXELS = int(config_json['FISH_AREA_MAX_PIXELS'])
            if 'FISH_HEIGHT_MIN_PIXELS' in config_json:
                FISH_HEIGHT_MIN_PIXELS = int(config_json['FISH_HEIGHT_MIN_PIXELS'])
            if 'FISH_HEIGHT_MAX_PIXELS' in config_json:
                FISH_HEIGHT_MAX_PIXELS = int(config_json['FISH_HEIGHT_MAX_PIXELS'])
            if 'FREEZE_CIRCLE_DIAMETER_PIXELS' in config_json:
                FREEZE_WINDOW_LEN = int(config_json['FREEZE_CIRCLE_DIAMETER_PIXELS'])
            if 'SECS_BEFORE_LOST' in config_json:
                SECS_B4_LOST = int(config_json['SECS_BEFORE_LOST'])
            if 'TRACKER_VIDEO_OUTPUT' in config_json:
                if config_json['TRACKER_VIDEO_OUTPUT'] == 'TRUE':
                    gen_tracker_video = True
                else:
                    gen_tracker_video = False
            if 'COWLOG_OUTPUT' in config_json:
                if config_json['COWLOG_OUTPUT'] == 'TRUE':
                    gen_cowlog = True
                else:
                    gen_cowlog = False
            if 'DELAY_TRACKING_SECS' in config_json:
                DELAY_TRACKING_SECS = int(config_json['DELAY_TRACKING_SECS'])
            if 'RULE1_SECS' in config_json:
                RULE1_SECS = int(config_json['RULE1_SECS'])
            if 'RULE2_SECS' in config_json:
                RULE2_SECS = int(config_json['RULE2_SECS'])
            if 'RULE3_SECS' in config_json:
                RULE3_SECS = int(config_json['RULE3_SECS'])
            if 'SCREEN_DELAY_SECS' in config_json:
                SCREEN_DELAY = int(config_json['SCREEN_DELAY_SECS'])
            if 'TESTING_CRITICAL_DUR_SECS' in config_json:
                TESTING_CRITICAL_DUR_SECS = int(config_json['TESTING_CRITICAL_DUR_SECS'])
            if 'TRAINING_CRITICAL_DUR_SECS' in config_json:
                TRAINING_CRITICAL_DUR_SECS = int(config_json['TRAINING_CRITICAL_DUR_SECS'])                
            if 'FEED_DELAY_SECS' in config_json:
                FEED_DELAY = int(config_json['FEED_DELAY_SECS'])
            if 'FEED_DURATION_SECS' in config_json:
                FEED_DURATION = int(config_json['FEED_DURATION_SECS'])
            if 'TOTAL_SECS' in config_json:
                TOTAL_TIME = int(config_json['TOTAL_SECS'])
            if 'NUM_FRAMES_FOR_BACKGROUND' in config_json:
                NUM_FRAMES_FOR_BACKGROUND = int(config_json['NUM_FRAMES_FOR_BACKGROUND'])
            if 'EDGE_BUFFER_PIXELS_X' in config_json:
                EDGE_BUFFER_X = int(config_json['EDGE_BUFFER_PIXELS_X'])
            if 'EDGE_BUFFER_PIXELS_Y' in config_json:
                EDGE_BUFFER_Y = int(config_json['EDGE_BUFFER_PIXELS_Y'])
            if 'PRE_SCREEN_EDGE_BUFFER_PIXELS_X' in config_json:
                PRE_SCREEN_EDGE_BUFFER_X = int(config_json['PRE_SCREEN_EDGE_BUFFER_PIXELS_X'])
            if 'PRE_SCREEN_EDGE_BUFFER_PIXELS_Y' in config_json:
                PRE_SCREEN_EDGE_BUFFER_Y = int(config_json['PRE_SCREEN_EDGE_BUFFER_PIXELS_Y'])
            if 'CROP_X1' in config_json:
                CROP_X1 = int(config_json['CROP_X1'])
            if 'CROP_X2' in config_json:
                CROP_X2 = int(config_json['CROP_X2'])
            if 'CROP_Y1' in config_json:
                CROP_Y1 = int(config_json['CROP_Y1'])
            if 'CROP_Y2' in config_json:
                CROP_Y2 = int(config_json['CROP_Y2'])
            #if 'TANK_LENGTH_CM' in config_json:
            #   TANK_LENGTH_CM = float(config_json['TANK_LENGTH_CM'])
            #if 'TANK_WIDTH_CM' in config_json:
            #   TANK_WIDTH_CM = float(config_json['TANK_WIDTH_CM'])
            #if 'MIRROR_LENGTH_CM' in config_json:
            #   MIRROR_LENGTH_CM = float(config_json['MIRROR_LENGTH_CM'])
            #if 'MIRROR_ZONE_WIDTH_CM' in config_json:
            #   MIRROR_ZONE_WIDTH_CM = int(config_json['MIRROR_ZONE_WIDTH_CM'])
            if 'TRACKING_WINDOW_LEN_PIXELS' in config_json:
                TRACKING_WINDOW_LEN = int(config_json['TRACKING_WINDOW_LEN_PIXELS'])
            #if 'TARGET_ZONE_CM' in config_json:
            #   TARGET_ZONE_CM = int(config_json['TARGET_ZONE_CM'])
            if 'HIGH_STIMULUS_LETTER' in config_json:
                HIGH_STIMULUS_LETTER = config_json['HIGH_STIMULUS_LETTER']
            if 'LOW_STIMULUS_LETTER' in config_json:
                LOW_STIMULUS_LETTER = config_json['LOW_STIMULUS_LETTER']
            if 'FREEZE_TIME_MIN_SECS' in config_json:
                FREEZE_TIME_MIN_SECS = int(config_json['FREEZE_TIME_MIN_SECS'])
            if fishid in config_json:
                tmp_dict = config_json[fishid]
                if 'TANK_UPPER_LEFT_X' in tmp_dict:
                    TANK_UPPER_LEFT_X = int(tmp_dict['TANK_UPPER_LEFT_X'])
                if 'TANK_UPPER_LEFT_Y' in tmp_dict:
                    TANK_UPPER_LEFT_Y = int(tmp_dict['TANK_UPPER_LEFT_Y'])
                if 'TANK_LOWER_LEFT_X' in tmp_dict:
                    TANK_LOWER_LEFT_X = int(tmp_dict['TANK_LOWER_LEFT_X'])
                if 'TANK_LOWER_LEFT_Y' in tmp_dict:
                    TANK_LOWER_LEFT_Y = int(tmp_dict['TANK_LOWER_LEFT_Y'])
                if 'TANK_UPPER_RIGHT_X' in tmp_dict:
                    TANK_UPPER_RIGHT_X = int(tmp_dict['TANK_UPPER_RIGHT_X'])
                if 'TANK_UPPER_RIGHT_Y' in tmp_dict:
                    TANK_UPPER_RIGHT_Y = int(tmp_dict['TANK_UPPER_RIGHT_Y'])
                if 'TANK_LOWER_RIGHT_X' in tmp_dict:
                    TANK_LOWER_RIGHT_X = int(tmp_dict['TANK_LOWER_RIGHT_X'])
                if 'TANK_LOWER_RIGHT_Y' in tmp_dict:
                    TANK_LOWER_RIGHT_Y = int(tmp_dict['TANK_LOWER_RIGHT_Y'])

        # print parameters
        print '\n', '#' * 45
        print 'TRACKER PARAMETERS: '
        print '#' * 45
        print 'gen_cowlog=' + str(gen_cowlog)
        print 'gen_tracker_video=' + str(gen_tracker_video)
        print 'DIFF_THRESHOLD=' + str(DIFF_THRESHOLD)
        print 'FISH_AREA_MIN_PIXELS=' + str(FISH_AREA_MIN_PIXELS)
        print 'FISH_AREA_MAX_PIXELS=' + str(FISH_AREA_MAX_PIXELS)
        print 'FISH_HEIGHT_MIN_PIXELS=' + str(FISH_HEIGHT_MIN_PIXELS)
        print 'FISH_HEIGHT_MAX_PIXELS=' + str(FISH_HEIGHT_MAX_PIXELS)
        print 'SECS_BEFORE_LOST=' + str(SECS_B4_LOST)
        print 'FEED_DELAY_SECS=' + str(FEED_DELAY)
        print 'SCREEN_DELAY_SECS=' + str(SCREEN_DELAY)
        print 'TESTING_CRITICAL_DUR_SECS=' + str(TESTING_CRITICAL_DUR_SECS)
        print 'TRAINING_CRITICAL_DUR_SECS=' + str(TRAINING_CRITICAL_DUR_SECS)
        print 'FEED_DURATION_SECS=' + str(FEED_DURATION)
        print 'TOTAL_SECS=' + str(TOTAL_TIME)
        print 'NUM_FRAMES_FOR_BACKGROUND=' + str(NUM_FRAMES_FOR_BACKGROUND)
        print 'EDGE_BUFFER_PIXELS_X=' + str(EDGE_BUFFER_X)
        print 'EDGE_BUFFER_PIXELS_Y=' + str(EDGE_BUFFER_Y)
        print 'PRE_SCREEN_EDGE_BUFFER_PIXELS_X=' + str(PRE_SCREEN_EDGE_BUFFER_X)
        print 'PRE_SCREEN_EDGE_BUFFER_PIXELS_Y=' + str(PRE_SCREEN_EDGE_BUFFER_Y)
        print 'CROP_X1=' + str(CROP_X1)
        print 'CROP_X2=' + str(CROP_X2)
        print 'CROP_Y1=' + str(CROP_Y1)
        print 'CROP_Y2=' + str(CROP_Y2)
        #print 'TANK_LENGTH_CM=' + str(TANK_LENGTH_CM)
        #print 'TANK_WIDTH_CM=' + str(TANK_WIDTH_CM)
        #print 'MIRROR_LENGTH_CM=' + str(MIRROR_LENGTH_CM)
        #print 'MIRROR_ZONE_WIDTH_CM=' + str(MIRROR_ZONE_WIDTH_CM)
        print 'FREEZE_CIRCLE_DIAMETER_PIXELS=' + str(FREEZE_WINDOW_LEN)
        print 'TRACKING_WINDOW_LEN_PIXELS=' + str(TRACKING_WINDOW_LEN)
        #print 'TARGET_ZONE_CM=' + str(TARGET_ZONE_CM)
        print 'HIGH_STIMULUS_LETTER=' + str(HIGH_STIMULUS_LETTER)
        print 'LOW_STIMULUS_LETTER=' + str(LOW_STIMULUS_LETTER)
        print 'FREEZE_TIME_MIN_SECS=' + str(FREEZE_TIME_MIN_SECS)
        print 'TANK_UPPER_LEFT_X=' + str(TANK_UPPER_LEFT_X)
        print 'TANK_UPPER_LEFT_Y=' + str(TANK_UPPER_LEFT_Y)
        print 'TANK_LOWER_LEFT_X=' + str(TANK_LOWER_LEFT_X)
        print 'TANK_LOWER_LEFT_Y=' + str(TANK_LOWER_LEFT_Y)
        print 'TANK_UPPER_RIGHT_X=' + str(TANK_UPPER_RIGHT_X)
        print 'TANK_UPPER_RIGHT_Y=' + str(TANK_UPPER_RIGHT_Y)
        print 'TANK_LOWER_RIGHT_X=' + str(TANK_LOWER_RIGHT_X)
        print 'TANK_LOWER_RIGHT_Y=' + str(TANK_LOWER_RIGHT_Y)
        print '#' * 45, '\n'

        # data to pull out
        reinforced_latency = 0
        non_reinforced_latency = 0
        time_in_reinforced_target = 0
        time_in_non_reinforced_target = 0
        time_in_mirror = 0
        prop_time_reinforced = 0.0
        prop_time_non_reinforced = 0.0
        prop_time_mirror = 0.0
        num_entries_reinforced = 0
        num_entries_non_reinforced = 0
        thigmotaxis_score = 0.0
        prop_time_thigmo = 0
        activity_level = 0
        prop_time_center = 0

        time_in_corners = 0
        prop_corners = 0

        # Data will be stored in a csv file
        csv_filename = 'numerosity_log.csv'
        freeze_log = 'num_freeze_log.csv'
        tracker_log = 'num_tracker_log.csv'
        
        if gen_cowlog:
            cowlog_filename = os.path.splitext(filename)[0] + '_cowlog.csv'
    
            # Open csv file in write mode and add header
            cowlog_file = open(cowlog_filename, 'w')
            cowlog_writer = csv.writer(cowlog_file)
            cowlog_writer.writerow(('time', 'code', 'class'))

        counter = 1
        faux_counter = 1
        trained_high = False
        trained_low = False
        leftside_high = False
        rightside_high = False
        
        first_target_zone = 'none'
        first_target_zone_entered = False
        left_target_frame_cnt = 0
        right_target_frame_cnt = 0
        left_target_latency_frame_cnt = 0
        left_target_entries = 0
        right_target_latency_frame_cnt = 0
        right_target_entries = 0
        
        prescreen_first_screen_zone = 'none'
        prescreen_first_screen_zone_entered = False
        prescreen_left_screen_frame_cnt = 0
        prescreen_right_screen_frame_cnt = 0
        prescreen_left_screen_latency_frame_cnt = 0
        prescreen_left_screen_entries = 0
        prescreen_right_screen_latency_frame_cnt = 0
        prescreen_right_screen_entries = 0
        
        postscreen_critical_first_screen_zone = 'none'
        postscreen_critical_first_screen_zone_entered = False
        postscreen_critical_left_screen_frame_cnt = 0
        postscreen_critical_right_screen_frame_cnt = 0
        postscreen_critical_left_screen_latency_frame_cnt = 0
        postscreen_critical_left_screen_entries = 0
        postscreen_critical_right_screen_latency_frame_cnt = 0
        postscreen_critical_right_screen_entries = 0
        end_of_prescreen_zone = 'unknown'
        start_of_postscreen_zone = 'unknown'
        prescreen_left_screen_latency_secs = 'NA'
        prescreen_right_screen_latency_secs = 'NA'
        postscreen_critical_left_screen_latency_secs = 'NA'
        postscreen_critical_right_screen_latency_secs = 'NA'
        
        last_known_zone = 'NA'
        in_left_target = False
        in_right_target = False
        in_upper_mirror = False
        in_lower_mirror = False
        thigmo_frame_cnt = 0
        mirror_frame_cnt = 0
        upper_mirror_frame_cnt = 0
        lower_mirror_frame_cnt = 0
        ul_thigmo_frame_cnt = 0
        ur_thigmo_frame_cnt = 0
        ll_thigmo_frame_cnt = 0
        lr_thigmo_frame_cnt = 0
        in_ul_thigmo = False
        in_ur_thigmo = False
        in_ll_thigmo = False
        in_lr_thigmo = False
        ul_corner_frame_cnt = 0
        ur_corner_frame_cnt = 0
        ll_corner_frame_cnt = 0
        lr_corner_frame_cnt = 0
        in_ul_corner = False
        in_ur_corner = False
        in_ll_corner = False
        in_lr_corner = False
        freeze_start = None
        freeze_zone = None
        freeze_event_cnt = 0
        freeze_frame_cnt = 0
        potential_freeze_frames = 0
        freeze_time_secs = 0
        total_time = 0
        time_in_center = 0
        in_center = False
        center_frame_cnt = 0

        # tracker metrics
        frames_b4_acq = 0
        acquired = False
        tracking = False
        frames_not_tracking = 0
        prescreen_frames_not_tracking = 0
        postscreen_critical_frames_not_tracking = 0
        lost_track_frame_cnt = 0
        times_lost_track = 0
        frames_since_last_track = []
        not_trackable = False

        print '#' * 45
        print 'Trial Parameters: '
        print '#' * 45
        print 'Species: ' + str(species)
        print 'Round: ' + str(round_num)
        print 'Standard Length: ' + str(std_len)
        print 'Sex: ' + str(sex)
        print 'Fish ID: ' + str(fishid)
        print 'Day: ' + str(day)
        print 'Session: ' + str(session)
        print 'Stimulus: ' + str(stimulus)
        print 'Other Stimulus: ' + str(that_stimulus)
        print 'Proportion: ' + str(proportion)
        print 'Fed side: ' + str(fedside)
        print 'Correct side: ' + str(correctside)
        print '#' * 45

        # determine trial type
        trial_type = 'NA'
        if int(day) < 3:
            trial_type = 'habituation'
            critical_dur_secs = 0
        elif int(day) < 9:
            trial_type = 'training'
            critical_dur_secs = TRAINING_CRITICAL_DUR_SECS
        elif fedside == 'none':
            trial_type = 'testing'
            critical_dur_secs = TESTING_CRITICAL_DUR_SECS
        else:
            trial_type = 'reinforce'
            critical_dur_secs = 0

        #print '!!!!'
        #print fishid[:1]
        #print '!!!!'
        # determine if fish was trained to high or low
        for ledda in HIGH_STIMULUS_LETTER:
            #print ledda
            if fishid[:1] == ledda:
                print 'Fish was trained to high stimulus!'
                trained_high = True
                trained_low  = False
        for ledda in LOW_STIMULUS_LETTER:
            #print ledda
            if fishid[:1] == ledda:
                print 'Fish was trained to low stimulus!'
                trained_high = False
                trained_low  = True
        if trained_high is False and trained_low is False:
            print 'ERROR> Unable to determine if fish was trained to high or low stimulus'
            print 'Fish ID should start with either ' + str(HIGH_STIMULUS_LETTER) + ' or ' + str(LOW_STIMULUS_LETTER)

        # determine if stimulus is high or low
        if stimulus.isdigit() and that_stimulus.isdigit():
            if int(stimulus) > int(that_stimulus):
                print 'Left side stimulus is high!'
                leftside_high = True
                rightside_high = False
            elif int(that_stimulus) > int(stimulus):
                print 'Right side stimulus is high'
                leftside_high = False
                rightside_high = True
            elif int(that_stimulus) == 0 and int(stimulus) == 0:
                print 'Both sides are blank screens, not a training or testing trial!'
                print 'For logging purposes, we will call right side the high side'
                leftside_high = False
                rightside_high = True
        else:
            print 'ERROR> One or both of the stimuli are not integer!'
            leftside_high = False
            rightside_high = False

        # length of a pixel (assuming square)
        #pixel_cm_len = TANK_LENGTH_CM/(TANK_UPPER_RIGHT_X-TANK_UPPER_LEFT_X)
        #print pixel_cm_len
        #pixel_cm_wid = TANK_WIDTH_CM/(TANK_LOWER_LEFT_Y-TANK_UPPER_RIGHT_Y)
        #print pixel_cm_wid
        #pixel_cm = (pixel_cm_len+pixel_cm_wid)/2
        #print pixel_cm

        # cropping window
        upper_bound, left_bound, right_bound, lower_bound = CROP_Y2, CROP_X1, CROP_X2, CROP_Y1
        new_upper_bound, new_left_bound, new_right_bound, new_lower_bound = CROP_Y2, CROP_X1, CROP_X2, CROP_Y1

        # target boxes
        #left_target_x = int(TANK_UPPER_LEFT_X + (TARGET_ZONE_CM * (1/pixel_cm)))
        #right_target_x = int(TANK_UPPER_RIGHT_X - (TARGET_ZONE_CM * (1/pixel_cm)))
        left_target_x = self.line6[0][0]
        left_target_y1 = self.line6[0][1]
        left_target_y2 = self.line6[1][1]
        right_target_x = self.line7[0][0]
        right_target_y1 = self.line7[0][1]
        right_target_y2 = self.line7[1][1]

        # target center point
        #left_target_center_x = ((left_target_x-TANK_LOWER_LEFT_X)/2) + TANK_LOWER_LEFT_X
        #target_center_y = ((TANK_LOWER_LEFT_Y-TANK_UPPER_LEFT_Y/2)) + TANK_UPPER_LEFT_Y
        #right_target_center_x = ((right_target_x-TANK_LOWER_RIGHT_X)/2) + TANK_LOWER_RIGHT_X

        # mirror boxes
        #upper_mirror_x1 = int((((TANK_UPPER_RIGHT_X-TANK_UPPER_LEFT_X)/2.0)+TANK_UPPER_LEFT_X) - ((MIRROR_LENGTH_CM/2.0) * (1/pixel_cm)))
        #upper_mirror_x2 = int((((TANK_UPPER_RIGHT_X-TANK_UPPER_LEFT_X)/2.0)+TANK_UPPER_LEFT_X) + ((MIRROR_LENGTH_CM/2.0) * (1/pixel_cm)))
        #upper_mirror_y = int(TANK_UPPER_LEFT_Y + (MIRROR_ZONE_WIDTH_CM * (1/pixel_cm)))
        upper_mirror_x1 = self.line11[0][0]
        upper_mirror_x2 = self.line12[0][0]
        upper_mirror_y = self.line11[1][1]


        #lower_mirror_x1 = int((((TANK_LOWER_RIGHT_X-TANK_LOWER_LEFT_X)/2.0)+TANK_LOWER_LEFT_X) - ((MIRROR_LENGTH_CM/2.0) * (1/pixel_cm)))
        #lower_mirror_x2 = int((((TANK_LOWER_RIGHT_X-TANK_LOWER_LEFT_X)/2.0)+TANK_LOWER_LEFT_X) + ((MIRROR_LENGTH_CM/2.0) * (1/pixel_cm)))
        #lower_mirror_y = int(TANK_LOWER_LEFT_Y - (MIRROR_ZONE_WIDTH_CM * (1/pixel_cm)))
        lower_mirror_x1 = self.line9[0][0]
        lower_mirror_x2 = self.line10[0][0]
        lower_mirror_y = self.line9[0][1]

        # thigmotaxis boxes
        #thigmo_ul_x1 = left_target_x + 1
        #thigmo_ul_x2 = upper_mirror_x1 - 1
        #thigmo_ur_x1 = upper_mirror_x2 + 1
        #thigmo_ur_x2 = right_target_x - 1
        #thigmo_upper_y = upper_mirror_y
        thigmo_ul_x1 = CROP_X1
        thigmo_ul_x2 = self.line11[0][0]
        thigmo_ur_x1 = self.line12[0][0]
        thigmo_ur_x2 = CROP_X2
        thigmo_upper_y = self.line11[1][1]


        #thigmo_ll_x1 = left_target_x + 1
        #thigmo_ll_x2 = lower_mirror_x1 - 1
        #thigmo_lr_x1 = lower_mirror_x2 + 1
        #thigmo_lr_x2 = right_target_x - 1
        #thigmo_lower_y = lower_mirror_y
        thigmo_ll_x1 = CROP_X1
        thigmo_ll_x2 = self.line9[0][0]
        thigmo_lr_x1 = self.line10[0][0]
        thigmo_lr_x2 = CROP_X2
        thigmo_lower_y = self.line9[0][1]

        # open the video
        #path = os.path.normpath(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print 'ERROR> Could not open :', path

        # get some info about the video
        length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        if length < 0:
            # manually count how many frames, this will take a little time...
            print 'Counting frames..'
            length = 0
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print 'ERROR> Could not open :', path
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

        # seconds per frame
        spf = 1.0/fps

        NUM_FRAMES_TO_SKIP = FEED_DELAY * fps
        NUM_FRAMES_TO_TRIM = (TOTAL_TIME-FEED_DURATION) * fps

        # Only process up to TOTAL_TIME
        if length > fps*TOTAL_TIME:
            length = fps*TOTAL_TIME
            
        # grab the 20th frame for drawing the rectangle
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print 'ERROR> Could not open :', path

        # calculate background image of tank for x frames
        background = get_background_image(cap,NUM_FRAMES_FOR_BACKGROUND,length, NUM_FRAMES_TO_SKIP, NUM_FRAMES_TO_TRIM)
        #background = get_background_image(cap,(length-NUM_FRAMES_TO_TRIM),length)

        # blur and crop background and save a copy of the background image for reference
        bm_initial = blur_and_mask(background, lower_bound, upper_bound, left_bound, right_bound, vidHeight, vidWidth)
        cv2.imwrite('background.jpg', background)

        startOfTrial = time.time()
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print 'ERROR> Could not open :', path
            sys.exit(1)

        first_pass = True
        prev_center = None

        print '\n\nProcessing...\n'
        while(cap.isOpened() and faux_counter*spf <= TOTAL_TIME):

            #print 'frame ' + str(counter) + '\n\n'
            if not faux_counter % fps:
                print 'Processing frame ' + str(faux_counter) + ' (' + str(faux_counter*spf) + ' secs)...'

            # for timing, maintaining constant fps
            beginningOfLoop = time.time()

            # skipping some frames at the beginning
            SKIPPY = 0
            if first_pass:
                for i in range(0, int(SKIPPY*fps)):
                    cap.read()
                    faux_counter += 1
                    frames_not_tracking += 1 # should this be counted?
                    
                    if i*spf <= SCREEN_DELAY:
                        prescreen_frames_not_tracking += 1
                    elif i*spf <= SCREEN_DELAY+critical_dur_secs:
                        postscreen_critical_frames_not_tracking += 1
                    
            ret,frame = cap.read()

            if ret == False:
                print 'ERROR> Did not read frame from video file'
                break

            if faux_counter == 1 and gen_tracker_video:
                video = cv2.VideoWriter(os.path.splitext(filename)[0] + '_tracker.avi',cv2.cv.CV_FOURCC(*'MJPG'),25,(vidWidth,vidHeight),True)

            if faux_counter > DELAY_TRACKING_SECS*fps:
                
                # Check Rule 1, if fish isn't detected in first RULE1_SECS then
                # declare video not trackable
                if (not acquired) and (faux_counter-(DELAY_TRACKING_SECS*fps) > (RULE1_SECS*fps)):
                    print "ERROR> Unable to track fish within first " + str(RULE1_SECS) + "secs"
                    print "Declaring video as not trackable"
                    not_trackable = True
                    break                
                
                # Check Rule 3, has track been lost for more than RULE3_SECS
                if (frames_not_tracking > (RULE3_SECS*fps)):
                    print "ERROR> Lost fish for more than " + str(RULE3_SECS) + "secs"
                    print "Declaring video as not trackable"
                    not_trackable = True
                    break
                
                # blur and crop frame
                bm = blur_and_mask(frame, lower_bound, upper_bound, left_bound, right_bound, vidHeight, vidWidth)
    
                # find difference between frame and background
                difference = cv2.absdiff(bm, bm_initial)
                #if counter < (FEED_DELAY*fps) or counter > ((FEED_DELAY+FEED_DURATION)*fps):
                #   #difference = apply_mask(difference, TANK_UPPER_LEFT_Y+EDGE_BUFFER, TANK_LOWER_LEFT_Y-EDGE_BUFFER, left_target_x, right_target_x, vidHeight, vidWidth)
                #   difference = apply_special_mask(difference, TANK_UPPER_LEFT_X+EDGE_BUFFER, TANK_UPPER_RIGHT_X-EDGE_BUFFER, TANK_UPPER_LEFT_Y+25, upper_mirror_y, left_target_x, right_target_x, lower_mirror_y, TANK_LOWER_LEFT_Y-25, vidHeight, vidWidth)
                #elif not tracking:
    
    
                #if faux_counter < (SCREEN_DELAY*fps):
                #   difference = apply_mask(difference, TANK_UPPER_LEFT_Y+PRE_SCREEN_EDGE_BUFFER_Y, TANK_LOWER_LEFT_Y-PRE_SCREEN_EDGE_BUFFER_Y, TANK_UPPER_LEFT_X+PRE_SCREEN_EDGE_BUFFER_X, TANK_UPPER_RIGHT_X-PRE_SCREEN_EDGE_BUFFER_X, vidHeight, vidWidth)
                #elif not tracking:
                #   difference = apply_mask(difference, TANK_UPPER_LEFT_Y+EDGE_BUFFER_Y, TANK_LOWER_LEFT_Y-EDGE_BUFFER_Y, TANK_UPPER_LEFT_X+EDGE_BUFFER_X, TANK_UPPER_RIGHT_X-EDGE_BUFFER_X, vidHeight, vidWidth)
                #else:
                #   difference = apply_mask(difference, new_lower_bound, new_upper_bound, new_left_bound, new_right_bound, vidHeight, vidWidth)
                difference = apply_test_mask(difference, upper_mirror_x1, upper_mirror_x2, TANK_UPPER_LEFT_Y+20, TANK_LOWER_LEFT_Y-20, TANK_UPPER_LEFT_X+20, TANK_UPPER_RIGHT_X-20, TANK_UPPER_LEFT_Y, TANK_LOWER_LEFT_Y, vidHeight, vidWidth)
    
                # find the centroid of the largest blob
                imdiff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
                if self.show_images:
                    #cv2.imshow('imdiff',imdiff)
                    cv2.imshow('imdiff', cv2.resize(imdiff, (0,0), fx=0.5, fy=0.5))
                #ret,thresh = cv2.threshold(imdiff,np.amax(imdiff),255,0)
                ret,thresh = cv2.threshold(imdiff,DIFF_THRESHOLD,255,0)
                if self.show_images:
                    #cv2.imshow('thresh',thresh)
                    cv2.imshow('thresh', cv2.resize(thresh, (0,0), fx=0.5, fy=0.5))
    
                #unified = merge_contours(frame, thresh)
                unified = None
                center = find_fish(thresh, vidWidth*vidHeight, unified, FISH_AREA_MIN_PIXELS, FISH_AREA_MAX_PIXELS, FISH_HEIGHT_MIN_PIXELS, FISH_HEIGHT_MAX_PIXELS)
                #center = None
                #print 'Center: ' + str(center) + '\n'
    
                # calc distance between current center and previous center
                if not first_pass:
                    if center is not None and prev_center is not None:
                        # distance between points in pixels times cm in a single pixel
                        #activity_level += (true_distance(prev_center, center))*pixel_cm
    
                        # this is just accumulating pixels traveled
                        activity_level += (true_distance(prev_center, center))
    
                # so we're tracking...
                if center is not None:
    
                    prev_center = center
    
                    if not acquired:
    
                        print 'Frames before acquisition: ' + str(frames_b4_acq)
    
                        acquired = True
    
                        # Thinking this is no longer issue for testing trials since
                        # the diff should be pretty clear even on first frame
                        if frames_b4_acq > 3.0*fps:
                            #dist_left = true_distance([left_target_center_x, target_center_y], center)
                            #dist_right = true_distance([right_target_center_x, target_center_y], center)
                            dist_left = 1
                            dist_right = 0
                            if dist_left > dist_right:
                                #right_target_frame_cnt += frames_b4_acq
                                right_target_entries = 1 #should this be counted
                                #print 'Counting first ' + str(frames_b4_acq) + ' frames to the right target zone'
                                first_target_zone = 'right'
                                first_target_zone_entered = True
                                
                                if 3.0*fps < SCREEN_DELAY:
                                    prescreen_right_screen_entries = 1
                                    prescreen_first_screen_zone = 'right'
                                    prescreen_first_screen_zone_entered = True
                                elif 3.0*fps < SCREEN_DELAY+critical_dur_secs:
                                    postscreen_critical_right_screen_entries = 1
                                    postscreen_critical_first_screen_zone = 'right'
                                    postscreen_critical_first_screen_zone_entered = True                              
                            else:
                                #left_target_frame_cnt += frames_b4_acq
                                left_target_entries = 1 #should this be counted
                                #print 'Counting first ' + str(frames_b4_acq) + ' frames to the left target zone'
                                first_target_zone = 'left'
                                first_target_zone_entered = True
                                
                                if 3.0*fps < SCREEN_DELAY:
                                    prescreen_left_screen_entries = 1
                                    prescreen_first_screen_zone = 'left'
                                    prescreen_first_screen_zone_entered = True                            
                                elif 3.0*fps < SCREEN_DELAY+critical_dur_secs:
                                    postscreen_critical_left_screen_entries = 1
                                    postscreen_critical_first_screen_zone = 'left'
                                    postscreen_critical_first_screen_zone_entered = True
    
                    if not tracking:
                        tracking = True
                        print 'Target acquired... tracking...'
                        print 'Frames since last tracking: ' + str(frames_not_tracking)
                        if acquired:
                            frames_since_last_track.append(frames_not_tracking)
    
                    # shrink search window
                    if center[1] - TRACKING_WINDOW_LEN < TANK_UPPER_LEFT_Y + EDGE_BUFFER_Y:
                        new_upper_bound = TANK_UPPER_LEFT_Y + EDGE_BUFFER_Y
                    else:
                        new_upper_bound = center[1] - TRACKING_WINDOW_LEN
                    if center[1] + TRACKING_WINDOW_LEN > TANK_LOWER_LEFT_Y - EDGE_BUFFER_Y:
                        new_lower_bound = TANK_LOWER_LEFT_Y - EDGE_BUFFER_Y
                    else:
                        new_lower_bound = center[1] + TRACKING_WINDOW_LEN
                    if center[0] - TRACKING_WINDOW_LEN < TANK_UPPER_LEFT_X + EDGE_BUFFER_X:
                        new_left_bound  = TANK_UPPER_LEFT_X + EDGE_BUFFER_X
                    else:
                        new_left_bound  = center[0] - TRACKING_WINDOW_LEN
                    if center[0] + TRACKING_WINDOW_LEN > TANK_UPPER_RIGHT_X - EDGE_BUFFER_X:
                        new_right_bound = TANK_UPPER_RIGHT_X - EDGE_BUFFER_X
                    else:
                        new_right_bound = center[0] + TRACKING_WINDOW_LEN
                    #print 'new_lower_bound=' + str(new_lower_bound)
                    #print 'new_upper_bound=' + str(new_upper_bound)
                    #print 'new_left_bound='  + str(new_left_bound)
                    #print 'new_right_bound=' + str(new_right_bound)
    
                    # if frames_not_tracking > 0 then apply half to prev zone and half to new zone
                    if in_left_target:
                        left_target_frame_cnt += (frames_not_tracking/2)
                    elif in_right_target:
                        right_target_frame_cnt += (frames_not_tracking/2)
                    elif in_lower_mirror:
                        lower_mirror_frame_cnt += (frames_not_tracking/2)
                    elif in_upper_mirror:
                        upper_mirror_frame_cnt += (frames_not_tracking/2)
                    elif in_upper_mirror:
                        upper_mirror_frame_cnt += (frames_not_tracking/2)
                    #elif in_ul_corner:
                    #   ul_corner_frame_cnt += (frames_not_tracking/2)
                    #elif in_ll_corner:
                    #   ll_corner_frame_cnt += (frames_not_tracking/2)
                    #elif in_ur_corner:
                    #   ur_corner_frame_cnt += (frames_not_tracking/2)
                    #elif in_lr_corner:
                    #   lr_corner_frame_cnt += (frames_not_tracking/2)
                    elif in_ul_thigmo:
                        ul_thigmo_frame_cnt += (frames_not_tracking/2)
                    elif in_ll_thigmo:
                        ll_thigmo_frame_cnt += (frames_not_tracking/2)
                    elif in_ur_thigmo:
                        ur_thigmo_frame_cnt += (frames_not_tracking/2)
                    elif in_lr_thigmo:
                        lr_thigmo_frame_cnt += (frames_not_tracking/2)
                    elif in_center:
                        center_frame_cnt += (frames_not_tracking/2)
    
                    frames_not_tracking = frames_not_tracking/2
                    
                    #if faux_counter*spf < SCREEN_DELAY:
                    #    prescreen_frames_not_tracking = prescreen_frames_not_tracking/2
                    #elif faux_counter*spf < SCREEN_DELAY+critical_dur_secs:
                    #    postscreen_critical_frames_not_tracking = postscreen_critical_frames_not_tracking/2
    
                    # check if fish is in left target
                    if center[0] < left_target_x and \
                    center[1] < left_target_y2 and \
                    center[1] > left_target_y1:
    
                        # check if left target zone entry occurred
                        if not in_left_target:
                            left_target_entries += 1
                            
                            if faux_counter*spf < SCREEN_DELAY:
                                prescreen_left_screen_entries += 1
                            elif faux_counter*spf < SCREEN_DELAY+critical_dur_secs:
                                postscreen_critical_left_screen_entries += 1
                                #print '!!!Postscreen Critical Left Target Entry!!!' + str(postscreen_critical_left_screen_entries)
    
                        if last_known_zone is not 'left screen' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'left screen', '1'))
    
                        left_target_frame_cnt += (frames_not_tracking + 1)
                        in_left_target = True
                        last_known_zone = 'left screen'
                        if not first_target_zone_entered:
                            first_target_zone = 'left'
                            first_target_zone_entered = True
                            
                        if faux_counter*spf < SCREEN_DELAY:
                            prescreen_left_screen_frame_cnt += (prescreen_frames_not_tracking + 1)
                            if not prescreen_first_screen_zone_entered:
                                prescreen_first_screen_zone = 'left'
                                prescreen_first_screen_zone_entered = True
                        elif faux_counter*spf < SCREEN_DELAY+critical_dur_secs:
                            postscreen_critical_left_screen_frame_cnt += (postscreen_critical_frames_not_tracking + 1)
                            if not postscreen_critical_first_screen_zone_entered:
                                postscreen_critical_first_screen_zone = 'left'
                                postscreen_critical_first_screen_zone_entered = True
                            
                    else:
                        in_left_target = False
    
                    # check if fish is in right target
                    if center[0] > right_target_x and \
                    center[1] < right_target_y2 and \
                    center[1] > right_target_y1:
    
                        # check if right target zone entry occurred
                        if not in_right_target:
                            right_target_entries += 1
                            
                            if faux_counter*spf < SCREEN_DELAY:
                                prescreen_right_screen_entries += 1
                            elif faux_counter*spf < SCREEN_DELAY+critical_dur_secs:
                                postscreen_critical_right_screen_entries += 1
                                #print '!!!Postscreen Critical Right Target Entry!!!' + str(postscreen_critical_right_screen_entries)
    
                        if last_known_zone is not 'right screen' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'right screen', '1'))
                            
                        right_target_frame_cnt += (frames_not_tracking + 1)
                        in_right_target = True
                        last_known_zone = 'right screen'
                        if not first_target_zone_entered:
                            first_target_zone = 'right'
                            first_target_zone_entered = True
                            
                        if faux_counter*spf < SCREEN_DELAY:
                            prescreen_right_screen_frame_cnt += (prescreen_frames_not_tracking + 1)
                            if not prescreen_first_screen_zone_entered:
                                prescreen_first_screen_zone = 'right'
                                prescreen_first_screen_zone_entered = True
                        elif faux_counter*spf < SCREEN_DELAY+critical_dur_secs:
                            postscreen_critical_right_screen_frame_cnt += (postscreen_critical_frames_not_tracking + 1)
                            if not postscreen_critical_first_screen_zone_entered:
                                postscreen_critical_first_screen_zone = 'right'
                                postscreen_critical_first_screen_zone_entered = True
                    else:
                        in_right_target = False
    
                    # check left target latency (if still applicable)
                    if left_target_entries == 0:
                        left_target_latency_frame_cnt += (frames_not_tracking + 1)
                        
                    if prescreen_left_screen_entries == 0 and \
                       faux_counter*spf < SCREEN_DELAY:
                        prescreen_left_screen_latency_frame_cnt += (prescreen_frames_not_tracking + 1)
                    elif postscreen_critical_left_screen_entries == 0 and \
                         faux_counter*spf < SCREEN_DELAY+critical_dur_secs and \
                         faux_counter*spf >= SCREEN_DELAY:
                        postscreen_critical_left_screen_latency_frame_cnt += (postscreen_critical_frames_not_tracking + 1)
    
                    # check right target latency (if still applicable)
                    if right_target_entries == 0:
                        right_target_latency_frame_cnt += (frames_not_tracking + 1)
                        
                    if prescreen_right_screen_entries == 0 and \
                       faux_counter*spf < SCREEN_DELAY:
                        prescreen_right_screen_latency_frame_cnt += (prescreen_frames_not_tracking + 1)
                    elif postscreen_critical_right_screen_entries == 0 and \
                         faux_counter*spf < SCREEN_DELAY+critical_dur_secs and \
                         faux_counter*spf >= SCREEN_DELAY:
                        postscreen_critical_right_screen_latency_frame_cnt += (postscreen_critical_frames_not_tracking + 1)
    
                    # check if in upper mirror zone
                    if center[0] > upper_mirror_x1 and center[0] < upper_mirror_x2 and center[1] < upper_mirror_y:
                        upper_mirror_frame_cnt += (frames_not_tracking + 1)
                            
                        # check if an entry to upper mirror occurred
                        if last_known_zone is not 'top mirror' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'top mirror', '1'))
                            
                        in_upper_mirror = True
                        last_known_zone = 'top mirror'
                    else:
                        in_upper_mirror = False
    
                    # check if in lower mirror zone
                    if center[0] > lower_mirror_x1 and center[0] < lower_mirror_x2 and center[1] > lower_mirror_y:
                        lower_mirror_frame_cnt += (frames_not_tracking + 1)
                        
                        # check if an entry to lower mirror occurred
                        if last_known_zone is not 'bottom mirror' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'bottom mirror', '1'))
    
                        in_lower_mirror = True
                        last_known_zone = 'bottom mirror'
                    else:
                        in_lower_mirror = False
    
                    # check if in upper left thigmotaxis zone
                    if center[0] > thigmo_ul_x1 and center[0] < thigmo_ul_x2 and center[1] < thigmo_upper_y:
                        ul_thigmo_frame_cnt += (frames_not_tracking + 1)
                        
                        # check if an entry to upper left thigmo occurred
                        if last_known_zone is not 'top left thigmo' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'top left thigmo', '1'))
                            
                        in_ul_thigmo = True
                        last_known_zone = 'top left thigmo'
                    else:
                        in_ul_thigmo = False
    
                    # check if in upper right thigmotaxis zone
                    if center[0] > thigmo_ur_x1 and center[0] < thigmo_ur_x2 and center[1] < thigmo_upper_y:
                        ur_thigmo_frame_cnt += (frames_not_tracking + 1)
                        
                        # check if an entry to upper right thigmo occurred
                        if last_known_zone is not 'top right thigmo' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'top right thigmo', '1'))
    
                        in_ur_thigmo = True
                        last_known_zone = 'top right thigmo'
                    else:
                        in_ur_thigmo = False
    
                    # check if in lower left thigmotaxis zone
                    if center[0] > thigmo_ll_x1 and center[0] < thigmo_ll_x2 and center[1] > thigmo_lower_y:
                        ll_thigmo_frame_cnt += (frames_not_tracking + 1)
                        
                        # check if an entry to lower left thigmo occurred
                        if last_known_zone is not 'bottom left thigmo' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'bottom left thigmo', '1'))
    
                        in_ll_thigmo = True
                        last_known_zone = 'bottom left thigmo'
                    else:
                        in_ll_thigmo = False
    
                    # check if in lower right thigmotaxis zone
                    if center[0] > thigmo_lr_x1 and center[0] < thigmo_lr_x2 and center[1] > thigmo_lower_y:
                        lr_thigmo_frame_cnt += (frames_not_tracking + 1)
                        
                        # check if an entry to lower right thigmo occurred
                        if last_known_zone is not 'bottom right thigmo' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'bottom right thigmo', '1'))
    
                        in_lr_thigmo = True
                        last_known_zone = 'bottom right thigmo'
                    else:
                        in_lr_thigmo = False
    
                    # check if fish is in corners (may use for thigmotaxis score)
    #               if center[0] < thigmo_ul_x1 and center[1] < thigmo_upper_y:
    #                   ul_corner_frame_cnt += (frames_not_tracking + 1)
    #                   in_ul_corner = True
    #               else:
    #                   in_ul_corner = False
    #
    #               if center[0] < thigmo_ll_x1 and center[1] > thigmo_lower_y:
    #                   ll_corner_frame_cnt += (frames_not_tracking + 1)
    #                   in_ll_corner = True
    #               else:
    #                   in_ll_corner = False
    #
    #               if center[0] > thigmo_ur_x2 and center[1] < thigmo_upper_y:
    #                   ur_corner_frame_cnt += (frames_not_tracking + 1)
    #                   in_ur_corner = True
    #               else:
    #                   in_ur_corner = False
    #
    #               if center[0] > thigmo_lr_x2 and center[1] > thigmo_lower_y:
    #                   lr_corner_frame_cnt += (frames_not_tracking + 1)
    #                   in_lr_corner = True
    #               else:
    #                   in_lr_corner = False
    
                    if not in_left_target and not in_right_target and not in_ll_corner and not in_lr_corner and not in_ul_corner and not in_ur_corner and not in_ll_thigmo and not in_lr_thigmo and not in_ul_thigmo and not in_ur_thigmo and not in_lower_mirror and not in_upper_mirror:
    
                        # check if an entry to center occurred
                        if last_known_zone is not 'center' and gen_cowlog:
                            # Add to cowlog
                            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'center', '1'))
    
                        in_center = True
                        last_known_zone = 'center'
                        center_frame_cnt += (frames_not_tracking + 1)
    
                    # check for freezing
                    if freeze_start is None:
                        freeze_start = center
                        if in_ll_thigmo or in_lr_thigmo or in_ul_thigmo or in_ur_thigmo:
                            freeze_zone = 'thigmo'
                        elif in_lower_mirror or in_upper_mirror:
                            freeze_zone = 'mirror'
                        elif in_left_target and (trained_high and leftside_high):
                            freeze_zone = 'reinforced.target'
                        elif in_left_target and (trained_low and rightside_high):
                            freeze_zone = 'non.reinforced.target'
                        elif in_right_target and (trained_high and rightside_high):
                            freeze_zone = 'reinforced.target'
                        elif in_right_target and (trained_low and leftside_high):
                            freeze_zone = 'non.reinforced.target'
                        else:
                            freeze_zone = 'center'
                    else:
                        if true_distance(freeze_start, center) < FREEZE_WINDOW_LEN:
                            freeze_frame_cnt += 1
    
                            # if potential freeze frames is not 0 then fish wasn't being tracked
                            # but appears to have remained within freeze circle so add those
                            # frames to the freeze frame counter
                            if potential_freeze_frames > 0:
                                freeze_frame_cnt += potential_freeze_frames
                                potential_freeze_frames = 0
                        else:
                            if freeze_frame_cnt > FREEZE_TIME_MIN_SECS*fps:
                                # log the freeze event
                                freeze_event_cnt += 1
                                print 'Freeze Event #' + str(freeze_event_cnt) + ': ' + str(freeze_frame_cnt*spf) + ' secs (' + freeze_zone + ')'
                                # Check if csv file exists
                                if not os.path.isfile(freeze_log):
                                    # Open csv file in write mode
                                    with open(freeze_log, 'w') as f:
                                        writer = csv.writer(f)
                                        # write the header
                                        writer.writerow(('Event','Fish.ID', 'Round', 'Day', \
                                                        'Session', 'Stimulus', 'Other.Stimulus', \
                                                        'Proportion', 'Fed.Side','Correct.Side', \
                                                        'Length.Secs','Zone'))
                                # Open csv file in append mode
                                with open(freeze_log, 'a') as f:
                                    writer = csv.writer(f)
                                    # write the data (limit all decimals to 2 digits)
                                    writer.writerow((freeze_event_cnt, fishid, round_num, day, session, stimulus,
                                                     that_stimulus, proportion, fedside, correctside, \
                                                     '{:.2f}'.format(freeze_frame_cnt*spf), freeze_zone))
                            freeze_frame_cnt = 0
                            freeze_start = center
                            if in_ll_thigmo or in_lr_thigmo or in_ul_thigmo or in_ur_thigmo:
                                freeze_zone = 'thigmo'
                            elif in_lower_mirror or in_upper_mirror:
                                freeze_zone = 'mirror'
                            elif in_left_target and (trained_high and leftside_high):
                                freeze_zone = 'reinforced.target'
                            elif in_left_target and (trained_low and rightside_high):
                                freeze_zone = 'non.reinforced.target'
                            elif in_right_target and (trained_high and rightside_high):
                                freeze_zone = 'reinforced.target'
                            elif in_right_target and (trained_low and leftside_high):
                                freeze_zone = 'non.reinforced.target'
                            else:
                                freeze_zone = 'center'
    
                    # draw red circle on largest
                    cv2.circle(frame,center,4,[0,0,255],-1)
    
                    frames_not_tracking = 0
                    postscreen_critical_frames_not_tracking = 0
                    prescreen_frames_not_tracking = 0
    
                else:
                    if frames_not_tracking > SECS_B4_LOST*fps:
                        # reset search window (last track)
                        if tracking:
                            print 'Lost track! Back to acquisition...'
                            times_lost_track += 1
    
                            # draw red circle on prev
                            if prev_center is not None:
                                cv2.circle(frame,prev_center,4,[0,0,255],-1)
    
                        new_upper_bound, new_left_bound, new_right_bound, new_lower_bound = CROP_Y2, CROP_X1, CROP_X2, CROP_Y1
    
                        lost_track_frame_cnt += 1
    
                        tracking = False
    
                    frames_not_tracking += 1                    
                    
                    if faux_counter*spf < SCREEN_DELAY:
                        prescreen_frames_not_tracking += 1
                    elif faux_counter*spf == SCREEN_DELAY and in_left_target:
                        prescreen_left_screen_frame_cnt += prescreen_frames_not_tracking
                        prescreen_frames_not_tracking = 0
                    elif faux_counter*spf == SCREEN_DELAY and in_right_target:
                        prescreen_right_screen_frame_cnt += prescreen_frames_not_tracking
                        prescreen_frames_not_tracking = 0
                    elif faux_counter*spf < SCREEN_DELAY+critical_dur_secs:
                        postscreen_critical_frames_not_tracking += 1
                    elif faux_counter*spf == SCREEN_DELAY+critical_dur_secs and in_left_target:
                        postscreen_critical_left_screen_frame_cnt += postscreen_critical_frames_not_tracking
                        postscreen_critical_frames_not_tracking = 0
                    elif faux_counter*spf == SCREEN_DELAY+critical_dur_secs and in_right_target:
                        postscreen_critical_right_screen_frame_cnt += postscreen_critical_frames_not_tracking
                        postscreen_critical_frames_not_tracking = 0
                        
                    if not acquired:
                        frames_b4_acq += 1
    
                    if acquired:
                        potential_freeze_frames += 1
                        
            else:
                if not faux_counter % fps:
                    print 'Not tracking yet...'

            # draw white box around search/track window
            if faux_counter < (SCREEN_DELAY*fps):
                cv2.rectangle(frame,(TANK_UPPER_LEFT_X+PRE_SCREEN_EDGE_BUFFER_X, TANK_LOWER_LEFT_Y-PRE_SCREEN_EDGE_BUFFER_Y), (TANK_UPPER_RIGHT_X-PRE_SCREEN_EDGE_BUFFER_X, TANK_UPPER_LEFT_Y+PRE_SCREEN_EDGE_BUFFER_Y), (255,255,255),2)
            elif not tracking:
                cv2.rectangle(frame,(TANK_UPPER_LEFT_X+EDGE_BUFFER_X, TANK_LOWER_LEFT_Y-EDGE_BUFFER_Y), (TANK_UPPER_RIGHT_X-EDGE_BUFFER_X, TANK_UPPER_LEFT_Y+EDGE_BUFFER_Y), (255,255,255),2)
            else:
                cv2.rectangle(frame,(new_left_bound, new_lower_bound), (new_right_bound, new_upper_bound), (255,255,255),2)

            # draw green circle around freeze center
            if freeze_start is not None:
                cv2.circle(frame,freeze_start,FREEZE_WINDOW_LEN,[0,255,0],2)

            # draw green box around target zones
            if in_left_target:
                cv2.rectangle(frame,(0,left_target_y1),(left_target_x, left_target_y2),(0,255,0),2)

            if in_right_target:
                cv2.rectangle(frame,(right_target_x,right_target_y1),(vidWidth,right_target_y2),(0,255,0),2)

            # draw blue box around mirror zones
            if in_upper_mirror:
                cv2.rectangle(frame,(upper_mirror_x1,0),(upper_mirror_x2,upper_mirror_y),(255,0,0),2)

            if in_lower_mirror:
                cv2.rectangle(frame,(lower_mirror_x1,lower_mirror_y),(lower_mirror_x2,vidHeight),(255,0,0),2)

            # draw red box around thigmo zones
            if in_ul_thigmo:
                cv2.rectangle(frame,(thigmo_ul_x1,0),(thigmo_ul_x2,thigmo_upper_y),(0,0,255),2)

            if in_ur_thigmo:
                cv2.rectangle(frame,(thigmo_ur_x1,0),(thigmo_ur_x2,thigmo_upper_y),(0,0,255),2)

            if in_ll_thigmo:
                cv2.rectangle(frame,(thigmo_ll_x1,thigmo_lower_y),(thigmo_ll_x2,vidHeight),(0,0,255),2)

            if in_lr_thigmo:
                cv2.rectangle(frame,(thigmo_lr_x1,thigmo_lower_y),(thigmo_lr_x2,vidHeight),(0,0,255),2)

            # switch to True if you want the grid drawn on every frame
            if False:
                cv2.rectangle(frame,(0,left_target_y1),(left_target_x,left_target_y2),(0,255,0),2)
                cv2.rectangle(frame,(right_target_x,right_target_y1),(vidWidth,right_target_y2),(0,255,0),2)
                cv2.rectangle(frame,(upper_mirror_x1,0),(upper_mirror_x2,upper_mirror_y),(255,0,0),2)
                cv2.rectangle(frame,(lower_mirror_x1,lower_mirror_y),(lower_mirror_x2,vidHeight),(255,0,0),2)
                cv2.rectangle(frame,(thigmo_ul_x1,0),(thigmo_ul_x2,thigmo_upper_y),(0,0,255),2)
                cv2.rectangle(frame,(thigmo_ur_x1,0),(thigmo_ur_x2,thigmo_upper_y),(0,0,255),2)
                cv2.rectangle(frame,(thigmo_ll_x1,thigmo_lower_y),(thigmo_ll_x2,vidHeight),(0,0,255),2)
                cv2.rectangle(frame,(thigmo_lr_x1,thigmo_lower_y),(thigmo_lr_x2,vidHeight),(0,0,255),2)

            if acquired and not (in_left_target or in_right_target or in_ll_corner or in_lr_corner or in_ul_corner or \
                                in_ur_corner or in_ll_thigmo or in_lr_thigmo or in_ul_thigmo or in_ur_thigmo or \
                                in_lower_mirror or in_upper_mirror):
                cv2.rectangle(frame,(left_target_x, thigmo_upper_y),(right_target_x,thigmo_lower_y),(255,255,0),2)
            # show frame
            if self.show_images:
                #cv2.imshow('image',frame)
                cv2.imshow('image', cv2.resize(frame, (0,0), fx=0.5, fy=0.5))
            
            if gen_tracker_video:
                video.write(frame)

            endOfLoop = time.time()

            #print 'time of loop: ' + str(round(time.time()-beginningOfLoop,4))

            k = cv2.waitKey(1)
            if k == 27:
                break

            if faux_counter*spf == SCREEN_DELAY - 1:
                if in_center:
                    end_of_prescreen_zone = 'center'
                elif in_left_target:
                    end_of_prescreen_zone = 'left screen'
                elif in_ll_thigmo:
                    end_of_prescreen_zone = 'lower left thigmo'
                elif in_lower_mirror:
                    end_of_prescreen_zone = 'lower mirror'
                elif in_lr_thigmo:
                    end_of_prescreen_zone = 'lower right thigmo'
                elif in_right_target:
                    end_of_prescreen_zone = 'right screen'
                elif in_ul_thigmo:
                    end_of_prescreen_zone = 'upper left thigmo'
                elif in_upper_mirror:
                    end_of_prescreen_zone = 'upper mirror'
                elif in_ur_thigmo:
                    end_of_prescreen_zone = 'upper right thigmo'
            elif faux_counter*spf == SCREEN_DELAY + 1:
                if in_center:
                    start_of_postscreen_zone = 'center'
                elif in_left_target:
                    start_of_postscreen_zone = 'left screen'
                elif in_ll_thigmo:
                    start_of_postscreen_zone = 'lower left thigmo'
                elif in_lower_mirror:
                    start_of_postscreen_zone = 'lower mirror'
                elif in_lr_thigmo:
                    start_of_postscreen_zone = 'lower right thigmo'
                elif in_right_target:
                    start_of_postscreen_zone = 'right screen'
                elif in_ul_thigmo:
                    start_of_postscreen_zone = 'upper left thigmo'
                elif in_upper_mirror:
                    start_of_postscreen_zone = 'upper mirror'
                elif in_ur_thigmo:
                    start_of_postscreen_zone = 'upper right thigmo'                
            
            if faux_counter*spf == SCREEN_DELAY:
                if prescreen_right_screen_entries > 0:
                    prescreen_right_screen_latency_secs = '{:.2f}'.format(prescreen_right_screen_latency_frame_cnt*spf)
                if prescreen_left_screen_entries > 0:
                    prescreen_left_screen_latency_secs = '{:.2f}'.format(prescreen_left_screen_latency_frame_cnt*spf)
            elif faux_counter*spf == SCREEN_DELAY+critical_dur_secs or faux_counter*spf == TOTAL_TIME:
                if postscreen_critical_right_screen_entries > 0:
                    postscreen_critical_right_screen_latency_secs = '{:.2f}'.format(postscreen_critical_right_screen_latency_frame_cnt*spf)
                if postscreen_critical_left_screen_entries > 0:
                    postscreen_critical_left_screen_latency_secs = '{:.2f}'.format(postscreen_critical_left_screen_latency_frame_cnt*spf)
                
            if acquired:
                counter+=1

            faux_counter+=1

            if first_pass:
                first_pass = False
                
        # Check Rule 2, if fish was lost last RULE2_SECS, then declare as not trackable
        if (not tracking) and (frames_not_tracking >= (RULE2_SECS*fps)):
            print "ERROR> Fish was not tracked for final " + str(RULE2_SECS) + "secs"
            print "Declaring video as not trackable"
            not_trackable = True            

        print '#' * 45
        print 'Trial Metrics: '
        print '#' * 45
        print 'Activity level: ' + str(activity_level) + ' cm'

        if leftside_high or rightside_high:
            if (trained_high and leftside_high) or (trained_low and not leftside_high):
                if left_target_entries == 0:
                    reinforced_latency = 'NA'
                else:
                    reinforced_latency = left_target_latency_frame_cnt * spf
                time_in_reinforced_target = left_target_frame_cnt * spf
                num_entries_reinforced = left_target_entries
                if left_target_frame_cnt == 0:
                    prop_time_reinforced = 0.0
                else:
                    prop_time_reinforced = (float(left_target_frame_cnt) / faux_counter) * 100.0
                if first_target_zone == 'left':
                    first_target_zone = 'Reinforced'
                elif first_target_zone == 'right':
                    first_target_zone = 'Non-Reinforced'
            elif (trained_high and rightside_high) or (trained_low and not rightside_high):
                if right_target_entries == 0:
                    reinforced_latency = 'NA'
                else:
                    reinforced_latency = right_target_latency_frame_cnt * spf
                time_in_reinforced_target = right_target_frame_cnt * spf
                num_entries_reinforced = right_target_entries
                if right_target_frame_cnt == 0:
                    prop_time_reinforced = 0.0
                else:
                    prop_time_reinforced = (float(right_target_frame_cnt) / faux_counter) * 100.0
                if first_target_zone == 'right':
                    first_target_zone = 'Reinforced'
                elif first_target_zone == 'left':
                    first_target_zone = 'Non-Reinforced'
            print 'Reinforced Target Zone Latency: ' + str(reinforced_latency) +  ' secs'
            print 'Time in Reinforced Target Zone: ' + str(time_in_reinforced_target) +  ' secs'
            print 'Number of Entries into Reinforced Target Zone: ' + str(num_entries_reinforced)
            print 'Proportion of time in Reinforced Target Zone: ' + str(prop_time_reinforced) + '%'
            print 'First target zone entered: ' + first_target_zone

            if (trained_low and leftside_high) or (trained_high and not leftside_high):
                if left_target_entries == 0:
                    non_reinforced_latency = 'NA'
                else:
                    non_reinforced_latency = left_target_latency_frame_cnt * spf
                time_in_non_reinforced_target = left_target_frame_cnt * spf
                num_entries_non_reinforced = left_target_entries
                if left_target_frame_cnt == 0:
                    prop_time_non_reinforced = 0.0
                else:
                    prop_time_non_reinforced = (float(left_target_frame_cnt) / faux_counter) * 100.0
            elif (trained_low and rightside_high) or (trained_high and not rightside_high):
                if right_target_entries == 0:
                    non_reinforced_latency = 'NA'
                else:
                    non_reinforced_latency = right_target_latency_frame_cnt * spf
                time_in_non_reinforced_target = right_target_frame_cnt * spf
                num_entries_non_reinforced = right_target_entries
                if right_target_frame_cnt == 0:
                    prop_time_non_reinforced = 0.0
                else:
                    prop_time_non_reinforced = (float(right_target_frame_cnt) / faux_counter) * 100.0
            print 'Non-Reinforced Target Zone Latency: ' + str(non_reinforced_latency) +  ' secs'
            print 'Time in Non-Reinforced Target Zone: ' + str(time_in_non_reinforced_target) +  ' secs'
            print 'Number of Entries into Non-Reinforced Target Zone: ' + str(num_entries_non_reinforced)
            print 'Proportion of time in Non-Reinforced Target Zone: ' + str(prop_time_non_reinforced) + '%'

        # Mirror zone metrics
        time_in_mirror = (upper_mirror_frame_cnt + lower_mirror_frame_cnt) * spf
        print 'Time in Mirror Zones: ' + str(time_in_mirror) + ' secs'
        if (upper_mirror_frame_cnt + lower_mirror_frame_cnt) == 0:
            prop_time_mirror = 0.0
        else:
            prop_time_mirror = (float(upper_mirror_frame_cnt + lower_mirror_frame_cnt) / faux_counter) * 100.0
        print 'Proportion time in Mirror Zones: ' + str(prop_time_mirror) + '%'

        # Thigmotaxis metrics
        thigmo_frame_cnt = ll_thigmo_frame_cnt + lr_thigmo_frame_cnt + ul_thigmo_frame_cnt + ur_thigmo_frame_cnt
        thigmotaxis_score = thigmo_frame_cnt * spf
        print 'Time in Thigmo Zones: ' + str(thigmotaxis_score) + ' secs'
        # TODO: is this correct for thigmotaxis score?

        if thigmo_frame_cnt == 0:
            prop_time_thigmo = 0.0
        else:
            prop_time_thigmo = (float(thigmo_frame_cnt) / faux_counter) * 100.0
        print 'Proportion time in Thigmotaxis Zones: ' + str(prop_time_thigmo) + '%'

        if (100 - (prop_time_reinforced+prop_time_non_reinforced+prop_time_mirror+prop_time_thigmo)) == 0:
            prop_time_center = 0.0
        else:
            prop_time_center = 100 - (prop_time_reinforced+prop_time_non_reinforced+prop_time_mirror+prop_time_thigmo)
        print 'Proportion time in Center Zone: ' + str(prop_time_center) + '%'

        total_time = faux_counter * spf
        time_in_center = total_time - (time_in_reinforced_target-time_in_non_reinforced_target-time_in_mirror-thigmotaxis_score)
        print 'Time in Center Zone: ' + str(time_in_center) + ' secs'

        time_in_corners = (ul_corner_frame_cnt + ur_corner_frame_cnt + ll_corner_frame_cnt + lr_corner_frame_cnt) * spf

        if (ul_corner_frame_cnt + ur_corner_frame_cnt + ll_corner_frame_cnt + lr_corner_frame_cnt) == 0:
            prop_corners = 0.0
        else:
            prop_corners = (float(ul_corner_frame_cnt + ur_corner_frame_cnt + ll_corner_frame_cnt + lr_corner_frame_cnt) / faux_counter) * 100.0
        print 'Time in Corners:' + str(time_in_corners) + ' secs'
        print 'Proportion time in Corners: ' + str(prop_corners) + '%'

        print 'Done Processing!'
        print '#' * 45
        print '\n\n'

        print '#' * 45
        print 'Write data out to ' + csv_filename

        # Check if csv file exists
        if not os.path.isfile(csv_filename):

            # Open csv file in write mode
            with open(csv_filename, 'w') as f:
                writer = csv.writer(f)

                # write the header
                writer.writerow(('Fish.ID', 'Round', 'Day', \
                                'Session', 'Stimulus', 'Other.Stimulus', \
                                'Proportion', 'Fed.Side','Correct.Side', 'Trial.Type', \
                                'Reinforced.Latency.Secs', 'Non.Reinforced.Latency.Secs', \
                                'Time.Reinforced.Secs', 'Time.Non.Reinforced.Secs', \
                                'Reinforced.Entries', 'Non.Reinforced.Entries', \
                                'Prop.Reinforced.%', 'Prop.Non.Reinforced.%', \
                                'Time.Mirror.Secs', 'Prop.Mirror.%', \
                                'Time.Thigmo.Secs', 'Prop.Thigmo.%', \
                                'Prop.Center.%', 'Activity.Level.Pixels', \
                                'Time.Corners.Secs', 'Prop.Corners.%', \
                                'First.Target.Zone', \
                                'Left.Target.Secs', 'Right.Target.Secs', \
                                'UL.Thigmo.Secs', 'LL.Thigmo.Secs', \
                                'UR.Thigmo.Secs', 'LR.Thigmo.Secs', \
                                'Center.Secs', 'Upper.Mirror.Secs', \
                                'Lower.Mirror.Secs', 'Total.Secs', \
                                'Prescreen.First.Screen.Zone', \
                                'Prescreen.Left.Screen.Latency.Secs', \
                                'Prescreen.Right.Screen.Latency.Secs', \
                                'Prescreen.Left.Screen.Time.Secs', \
                                'Prescreen.Right.Screen.Time.Secs', \
                                'Prescreen.Left.Screen.Entries', \
                                'Prescreen.Right.Screen.Entries', \
                                'Postscreen.Critical.First.Screen.Zone', \
                                'Postscreen.Critical.Left.Screen.Latency.Secs', \
                                'Postscreen.Critical.Right.Screen.Latency.Secs', \
                                'Postscreen.Critical.Left.Screen.Time.Secs', \
                                'Postscreen.Critical.Right.Screen.Time.Secs', \
                                'Postscreen.Critical.Left.Screen.Entries', \
                                'Postscreen.Critical.Right.Screen.Entries', \
                                'End.Of.Prescreen.Zone', \
                                'Start.Of.Postscreen.Zone'))

        # Open csv file in append mode
        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)

            if not_trackable:
                # notify user that video was not trackable
                writer.writerow((fishid, round_num, day, session, stimulus, that_stimulus, \
                                 proportion, fedside, correctside, trial_type, "Not Trackable"))                
            else:
                # write the data (limit all decimals to 2 digits)
                writer.writerow((fishid, round_num, day, session, stimulus, that_stimulus, \
                                 proportion, fedside, correctside, trial_type, \
                                 reinforced_latency, non_reinforced_latency, \
                                 '{:.2f}'.format(time_in_reinforced_target), '{:.2f}'.format(time_in_non_reinforced_target), \
                                 num_entries_reinforced, num_entries_non_reinforced, \
                                 '{:.2f}'.format(prop_time_reinforced), '{:.2f}'.format(prop_time_non_reinforced), \
                                 '{:.2f}'.format(time_in_mirror), '{:.2f}'.format(prop_time_mirror), \
                                 '{:.2f}'.format(thigmotaxis_score), '{:.2f}'.format(prop_time_thigmo), \
                                 '{:.2f}'.format(prop_time_center), '{:.2f}'.format(activity_level), \
                                 '{:.2f}'.format(time_in_corners), '{:.2f}'.format(prop_corners), \
                                 first_target_zone, \
                                 '{:.2f}'.format((left_target_frame_cnt-ll_corner_frame_cnt-ul_corner_frame_cnt)*spf), \
                                 '{:.2f}'.format((right_target_frame_cnt-lr_corner_frame_cnt-ur_corner_frame_cnt)*spf), \
                                 '{:.2f}'.format((ul_thigmo_frame_cnt+ul_corner_frame_cnt)*spf), \
                                 '{:.2f}'.format((ll_thigmo_frame_cnt+ll_corner_frame_cnt)*spf), \
                                 '{:.2f}'.format((ur_thigmo_frame_cnt+ur_corner_frame_cnt)*spf), \
                                 '{:.2f}'.format((lr_thigmo_frame_cnt+lr_corner_frame_cnt)*spf), \
                                 '{:.2f}'.format(center_frame_cnt*spf), \
                                 '{:.2f}'.format(upper_mirror_frame_cnt*spf), \
                                 '{:.2f}'.format(lower_mirror_frame_cnt*spf), \
                                 '{:.2f}'.format((left_target_frame_cnt+right_target_frame_cnt+ul_thigmo_frame_cnt+ll_thigmo_frame_cnt+ur_thigmo_frame_cnt+lr_thigmo_frame_cnt+center_frame_cnt+lower_mirror_frame_cnt+upper_mirror_frame_cnt)*spf), \
                                 prescreen_first_screen_zone, \
                                 prescreen_left_screen_latency_secs, \
                                 prescreen_right_screen_latency_secs, \
                                 '{:.2f}'.format(prescreen_left_screen_frame_cnt*spf), \
                                 '{:.2f}'.format(prescreen_right_screen_frame_cnt*spf), \
                                 prescreen_left_screen_entries, prescreen_right_screen_entries, \
                                 postscreen_critical_first_screen_zone, \
                                 postscreen_critical_left_screen_latency_secs, \
                                 postscreen_critical_right_screen_latency_secs, \
                                 '{:.2f}'.format(postscreen_critical_left_screen_frame_cnt*spf), \
                                 '{:.2f}'.format(postscreen_critical_right_screen_frame_cnt*spf), \
                                 postscreen_critical_left_screen_entries, postscreen_critical_right_screen_entries, \
                                 end_of_prescreen_zone, start_of_postscreen_zone))
        print 'Done with ' + csv_filename + '!'
        print '#' * 45

        print '\n'
        print '#' * 45
        print 'Write tracker stats out to ' + tracker_log

        # Check if csv file exists
        if not os.path.isfile(tracker_log):

            # Open csv file in write mode
            with open(tracker_log, 'w') as f:
                writer = csv.writer(f)

                # write the header
                writer.writerow(('File', 'Total.Frames', 'Total.Time', \
                                'Lost.Track.Frame.Cnt', \
                                'Times.Lost.Track', 'Frames.Before.Acq', \
                                'Frames.Btwn.Tracks'))

        # Open csv file in append mode
        with open(tracker_log, 'a') as f:
            writer = csv.writer(f)

            # write the data (limit all decimals to 2 digits)
            writer.writerow((filename, faux_counter,  '{:.2f}'.format(faux_counter*spf), \
                            lost_track_frame_cnt, times_lost_track, \
                            frames_b4_acq, frames_since_last_track))
        print 'Done with ' + tracker_log + '!'
        print '#' * 45

        ### after the program exits, print some useful stuff to the screen
        # first calculate realized fps
        print '#' * 45
        print 'Some Useful stuff...'
        print 'Frame count: ' + str(faux_counter)
        print 'Total Video Time: ' + str(faux_counter*spf) + ' secs'
        print 'Lost track frame count: ' + str(lost_track_frame_cnt)
        print 'Times lost track: ' + str(times_lost_track)
        print 'Frames before acquisition: ' + str(frames_b4_acq)
        print '\nThis program took ' + str(time.time() - start_time) + " seconds to run."
        print '#' * 45
        
        if gen_cowlog:
            # write last line of cowlog
            cowlog_writer.writerow(('{:.2f}'.format(faux_counter*spf), 'END', '0'))

            # close the cowlog file
            cowlog_file.close()
        
        cv2.destroyAllWindows()
        
        if gen_tracker_video:
            video.release()

if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path2video', help = 'Path to video file including the filename', required=True)
    ap.add_argument('-c', '--config_file', help = 'Path to config file (json) including the filename', required=True)
    ap.add_argument('-f', '--fish_json', help = 'Path to fish.json including the filename', required=True)
    ap.add_argument('-g', '--grid_json', help = 'Path to numerosity_grid.json including the filename', required=True)
    ap.add_argument('-s', '--show_images', help='Whether or not show images (slows processing)',action='store_true')
    args = vars(ap.parse_args())

    path = args['path2video']

    #path = r'D:\num\gambusia_17_201_male_winston_9_1_7_14_50_none_R.mp4'
    #path = r'D:\num\gambusia_17_376_female_glinda_9_1_7_14_50_none_L.mp4'
    #path = r'E:\new_num\gambusia_17_355_female_winnie_9_1_9_12_75_none_R.mp4'
    #path = r'D:\new_num\gambusia_18_TBD_female_Gail_9_1_9_12_75_none_L.mp4'
    #path = r'D:\new_num\gambusia_17_373_female_grace_9_1_9_12_75_none_L.mp4'
    #path = r'D:\new_num\gambusia_18_TBD_female_Willow_9_1_9_12_75_none_R.mp4'
    #path = r'D:\new_num\gambusia_17_234_male_gregory_9_1_9_12_75_none_L.mp4'
    #path = r'D:\new_num\gambusia_18_TBD_male_George_9_1_9_12_75_none_L.mp4'
    #path = r'D:\new_num\gambusia_17_200_male_walter_9_1_8_12_67_none_R.mp4'
    #path = r'D:\new_num\gambusia_18_TBD_female_Wendy_9_1_9_12_75_none_R.mp4'
    #path = r'D:\new_num\gambusia_18_TBD_male_Wheatley_9_1_9_12_75_none_R.mp4'
    #path = r'D:\new_num\gambusia_18_TBD_male_Gary_9_1_9_12_75_none_L.mp4'

    vid_file = str(path)
    config_file = str(args['config_file'])
    fish_json = str(args['fish_json'])
    grid_json = str(args['grid_json'])

    if args['show_images']:
        show_images = True
    else:
        show_images = False

    this_tracker = Tracker(vid_file, config_file, fish_json, grid_json, show_images)
    this_tracker.run_numerosity_tracker()
