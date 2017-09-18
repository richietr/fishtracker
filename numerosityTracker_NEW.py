import numpy as np
import cv2
import sys
import time
import math
import os
import csv
#import matplotlib.pyplot as plt
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

	global cant_decide

	# find all contours in the frame
	contours = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
			if area < max_area_pixels*1.0 and area > min_area_pixels*1.0 and max(h,w) < max_height_pixels and max(h,w) > min_height_pixels:
				idx = area_list.index(area)
				potential_tracks.append(contours[idx])
			#else:
				#print 'Area=' + str(area) + ', MaxH=' + str(max(h,w))


		#print 'Number of potential tracks: ' + str(len(potential_tracks)) + ' of ' + str(len(area_list))

		if len(potential_tracks) > 1 or len(potential_tracks) == 0:
			cant_decide += 1
			return None
		else:

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

	global cant_decide
	cant_decide = 0

	def __init__(self, vid_file, config_file, show_images):

		self.vid_file = vid_file
		self.config_file = config_file
		self.show_images = show_images

		print 'Video file: ' + str(vid_file)
		print 'Config file (json): ' + str(config_file)

		if show_images:
			print 'Showing images!'
		else:
			print 'NOT showing images!'


	def run_numerosity_tracker(self):

		print time.strftime('%X %x %Z')
		start_time = time.time()

		path = self.vid_file

		# Parse out filepath
		path_strip = os.path.splitext(path)[0]
		#path_parts = path_strip.split('/')
		path_parts = path_strip.split('\\')
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
		NUM_FRAMES_FOR_BACKGROUND = 1000
		PRE_SCREEN_EDGE_BUFFER_X = 175 #pixels
		PRE_SCREEN_EDGE_BUFFER_Y = 175 #pixels
		EDGE_BUFFER_X = 75 #pixels
		EDGE_BUFFER_Y = 75 #pixels
		CROP_X1 = 0
		CROP_X2 = 1296
		CROP_Y1 = 0
		CROP_Y2 = 972
		TANK_LENGTH_CM = 40.0 #40.64
		TANK_WIDTH_CM = 20.0 #21.59
		TANK_UPPER_LEFT_X = 0
		TANK_UPPER_LEFT_Y = 0
		TANK_LOWER_LEFT_X = 0
		TANK_LOWER_LEFT_Y = 972
		TANK_UPPER_RIGHT_X = 1296
		TANK_UPPER_RIGHT_Y = 0
		TANK_LOWER_RIGHT_X = 1296
		TANK_LOWER_RIGHT_Y = 972
		MIRROR_LENGTH_CM = 5.08 # 2"
		MIRROR_ZONE_WIDTH_CM = 3
		TRACKING_WINDOW_LEN = 100 # TODO: Use cm instead of pixels
		TARGET_ZONE_CM = 11
		HIGH_STIMULUS_LETTER = 'I'
		LOW_STIMULUS_LETTER = 'O'
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
			if 'SCREEN_DELAY_SECS' in config_json:
				SCREEN_DELAY = int(config_json['SCREEN_DELAY_SECS'])
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
			if 'TANK_LENGTH_CM' in config_json:
				TANK_LENGTH_CM = float(config_json['TANK_LENGTH_CM'])
			if 'TANK_WIDTH_CM' in config_json:
				TANK_WIDTH_CM = float(config_json['TANK_WIDTH_CM'])
			if 'MIRROR_LENGTH_CM' in config_json:
				MIRROR_LENGTH_CM = float(config_json['MIRROR_LENGTH_CM'])
			if 'MIRROR_ZONE_WIDTH_CM' in config_json:
				MIRROR_ZONE_WIDTH_CM = int(config_json['MIRROR_ZONE_WIDTH_CM'])
			if 'TRACKING_WINDOW_LEN_PIXELS' in config_json:
				TRACKING_WINDOW_LEN = int(config_json['TRACKING_WINDOW_LEN_PIXELS'])
			if 'TARGET_ZONE_CM' in config_json:
				TARGET_ZONE_CM = int(config_json['TARGET_ZONE_CM'])
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
		print 'DIFF_THRESHOLD=' + str(DIFF_THRESHOLD)
		print 'FISH_AREA_MIN_PIXELS=' + str(FISH_AREA_MIN_PIXELS)
		print 'FISH_AREA_MAX_PIXELS=' + str(FISH_AREA_MAX_PIXELS)
		print 'FISH_HEIGHT_MIN_PIXELS=' + str(FISH_HEIGHT_MIN_PIXELS)
		print 'FISH_HEIGHT_MAX_PIXELS=' + str(FISH_HEIGHT_MAX_PIXELS)
		print 'SECS_BEFORE_LOST=' + str(SECS_B4_LOST)
		print 'FEED_DELAY_SECS=' + str(FEED_DELAY)
		print 'SCREEN_DELAY_SECS=' + str(SCREEN_DELAY)
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
		print 'TANK_LENGTH_CM=' + str(TANK_LENGTH_CM)
		print 'TANK_WIDTH_CM=' + str(TANK_WIDTH_CM)
		print 'MIRROR_LENGTH_CM=' + str(MIRROR_LENGTH_CM)
		print 'MIRROR_ZONE_WIDTH_CM=' + str(MIRROR_ZONE_WIDTH_CM)
		print 'FREEZE_CIRCLE_DIAMETER_PIXELS=' + str(FREEZE_WINDOW_LEN)
		print 'TRACKING_WINDOW_LEN_PIXELS=' + str(TRACKING_WINDOW_LEN)
		print 'TARGET_ZONE_CM=' + str(TARGET_ZONE_CM)
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
		#zone_log = 'zone_log.csv'

		counter = 0
		faux_counter = 0
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
		lost_track_frame_cnt = 0
		times_lost_track = 0
		frames_since_last_track = []

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
		elif int(day) < 9:
			trial_type = 'training'
		elif fedside == 'none':
			trial_type = 'testing'
		else:
			trial_type = 'reinforce'

		# determine if fish was trained to high or low
		if fishid.startswith(HIGH_STIMULUS_LETTER):
			print 'Fish was trained to high stimulus!'
			trained_high = True
			trained_low  = False
		elif fishid.startswith(LOW_STIMULUS_LETTER):
			print 'Fish was trained to low stimulus!'
			trained_high = False
			trained_low  = True
		else:
			print 'ERROR> Unable to determine if fish was trained to high or low stimulus'
			print 'Fish ID should start with either ' + HIGH_STIMULUS_LETTER + ' or ' + LOW_STIMULUS_LETTER
			trained_high = False
			trained_low  = False

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
		pixel_cm_len = TANK_LENGTH_CM/(TANK_UPPER_RIGHT_X-TANK_UPPER_LEFT_X)
		#print pixel_cm_len
		pixel_cm_wid = TANK_WIDTH_CM/(TANK_LOWER_LEFT_Y-TANK_UPPER_RIGHT_Y)
		#print pixel_cm_wid
		pixel_cm = (pixel_cm_len+pixel_cm_wid)/2
		#print pixel_cm

		# cropping window
		upper_bound, left_bound, right_bound, lower_bound = CROP_Y2, CROP_X1, CROP_X2, CROP_Y1
		new_upper_bound, new_left_bound, new_right_bound, new_lower_bound = CROP_Y2, CROP_X1, CROP_X2, CROP_Y1

		# target boxes
		left_target_x = int(TANK_UPPER_LEFT_X + (TARGET_ZONE_CM * (1/pixel_cm)))
		right_target_x = int(TANK_UPPER_RIGHT_X - (TARGET_ZONE_CM * (1/pixel_cm)))

		# target center point
		left_target_center_x = ((left_target_x-TANK_LOWER_LEFT_X)/2) + TANK_LOWER_LEFT_X
		target_center_y = ((TANK_LOWER_LEFT_Y-TANK_UPPER_LEFT_Y/2)) + TANK_UPPER_LEFT_Y
		right_target_center_x = ((right_target_x-TANK_LOWER_RIGHT_X)/2) + TANK_LOWER_RIGHT_X

		# mirror boxes
		upper_mirror_x1 = int((((TANK_UPPER_RIGHT_X-TANK_UPPER_LEFT_X)/2.0)+TANK_UPPER_LEFT_X) - ((MIRROR_LENGTH_CM/2.0) * (1/pixel_cm)))
		upper_mirror_x2 = int((((TANK_UPPER_RIGHT_X-TANK_UPPER_LEFT_X)/2.0)+TANK_UPPER_LEFT_X) + ((MIRROR_LENGTH_CM/2.0) * (1/pixel_cm)))
		upper_mirror_y = int(TANK_UPPER_LEFT_Y + (MIRROR_ZONE_WIDTH_CM * (1/pixel_cm)))

		lower_mirror_x1 = int((((TANK_LOWER_RIGHT_X-TANK_LOWER_LEFT_X)/2.0)+TANK_LOWER_LEFT_X) - ((MIRROR_LENGTH_CM/2.0) * (1/pixel_cm)))
		lower_mirror_x2 = int((((TANK_LOWER_RIGHT_X-TANK_LOWER_LEFT_X)/2.0)+TANK_LOWER_LEFT_X) + ((MIRROR_LENGTH_CM/2.0) * (1/pixel_cm)))
		lower_mirror_y = int(TANK_LOWER_LEFT_Y - (MIRROR_ZONE_WIDTH_CM * (1/pixel_cm)))

		# thigmotaxis boxes
		thigmo_ul_x1 = left_target_x + 1
		thigmo_ul_x2 = upper_mirror_x1 - 1
		thigmo_ur_x1 = upper_mirror_x2 + 1
		thigmo_ur_x2 = right_target_x - 1
		thigmo_upper_y = upper_mirror_y

		thigmo_ll_x1 = left_target_x + 1
		thigmo_ll_x2 = lower_mirror_x1 - 1
		thigmo_lr_x1 = lower_mirror_x2 + 1
		thigmo_lr_x2 = right_target_x - 1
		thigmo_lower_y = lower_mirror_y

		# open the video
		#path = os.path.normpath(path)
		cap = cv2.VideoCapture(path)
		if not cap.isOpened():
			print 'ERROR> Could not open :',path

		# get some info about the video
		length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		#length = 5781
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
		while(cap.isOpened()):

			#print 'frame ' + str(counter) + '\n\n'
			if not faux_counter % 100:
				print 'Processing frame ' + str(faux_counter) + '...'

			# for timing, maintaining constant fps
			beginningOfLoop = time.time()

			# skipping some frames at the beginning
			if first_pass:
				for i in range(0, int(30*fps)):
					cap.read()
					frames_not_tracking += 1 # should this be counted?
			ret,frame = cap.read()

			if ret == False:
				print 'ERROR> Did not read frame from video file'
				break

			# blur and crop frame
			bm = blur_and_mask(frame, lower_bound, upper_bound, left_bound, right_bound, vidHeight, vidWidth)

			# find difference between frame and background
			difference = cv2.absdiff(bm, bm_initial)
			#if counter < (FEED_DELAY*fps) or counter > ((FEED_DELAY+FEED_DURATION)*fps):
			#	#difference = apply_mask(difference, TANK_UPPER_LEFT_Y+EDGE_BUFFER, TANK_LOWER_LEFT_Y-EDGE_BUFFER, left_target_x, right_target_x, vidHeight, vidWidth)
			#	difference = apply_special_mask(difference, TANK_UPPER_LEFT_X+EDGE_BUFFER, TANK_UPPER_RIGHT_X-EDGE_BUFFER, TANK_UPPER_LEFT_Y+25, upper_mirror_y, left_target_x, right_target_x, lower_mirror_y, TANK_LOWER_LEFT_Y-25, vidHeight, vidWidth)
			#elif not tracking:
			if faux_counter < (SCREEN_DELAY*fps):
				difference = apply_mask(difference, TANK_UPPER_LEFT_Y+PRE_SCREEN_EDGE_BUFFER_Y, TANK_LOWER_LEFT_Y-PRE_SCREEN_EDGE_BUFFER_Y, TANK_UPPER_LEFT_X+PRE_SCREEN_EDGE_BUFFER_X, TANK_UPPER_RIGHT_X-PRE_SCREEN_EDGE_BUFFER_X, vidHeight, vidWidth)
			elif not tracking:
				difference = apply_mask(difference, TANK_UPPER_LEFT_Y+EDGE_BUFFER_Y, TANK_LOWER_LEFT_Y-EDGE_BUFFER_Y, TANK_UPPER_LEFT_X+EDGE_BUFFER_X, TANK_UPPER_RIGHT_X-EDGE_BUFFER_X, vidHeight, vidWidth)
			else:
				difference = apply_mask(difference, new_lower_bound, new_upper_bound, new_left_bound, new_right_bound, vidHeight, vidWidth)

			# find the centroid of the largest blob
			imdiff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
			if show_images:
				#cv2.imshow('imdiff',imdiff)
				cv2.imshow('imdiff', cv2.resize(imdiff, (0,0), fx=0.5, fy=0.5))
			#ret,thresh = cv2.threshold(imdiff,np.amax(imdiff),255,0)
			ret,thresh = cv2.threshold(imdiff,DIFF_THRESHOLD,255,0)
			if show_images:
				#cv2.imshow('thresh',thresh)
				cv2.imshow('thresh', cv2.resize(thresh, (0,0), fx=0.5, fy=0.5))

			unified = merge_contours(frame, thresh)
			#unified = None
			center = find_fish(thresh, vidWidth*vidHeight, unified, FISH_AREA_MIN_PIXELS, FISH_AREA_MAX_PIXELS, FISH_HEIGHT_MIN_PIXELS, FISH_HEIGHT_MAX_PIXELS)
			#center = None
			#print 'Center: ' + str(center) + '\n'

			# calc distance between current center and previous center
			if not first_pass:
				if center is not None and prev_center is not None:
					# distance between points in pixels times cm in a single pixel
					activity_level += (true_distance(prev_center, center))*pixel_cm

			if center is not None:

				prev_center = center

				if not acquired:
					acquired = True
					dist_left = true_distance([left_target_center_x, target_center_y], center)
					dist_right = true_distance([right_target_center_x, target_center_y], center)
					if dist_left > dist_right:
						#right_target_frame_cnt += frames_b4_acq
						right_target_entries = 1 #should this be counted
						#print 'Counting first ' + str(frames_b4_acq) + ' frames to the right target zone'
						first_target_zone = 'right'
						first_target_zone_entered = True
					else:
						#left_target_frame_cnt += frames_b4_acq
						left_target_entries = 1 #should this be counted
						#print 'Counting first ' + str(frames_b4_acq) + ' frames to the left target zone'
						first_target_zone = 'left'
						first_target_zone_entered = True

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
				elif in_ul_corner:
					ul_corner_frame_cnt += (frames_not_tracking/2)
				elif in_ll_corner:
					ll_corner_frame_cnt += (frames_not_tracking/2)
				elif in_ur_corner:
					ur_corner_frame_cnt += (frames_not_tracking/2)
				elif in_lr_corner:
					lr_corner_frame_cnt += (frames_not_tracking/2)
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

				# check if left target zone entry occurred
				if not in_left_target and center[0] < left_target_x:
					left_target_entries += 1

				# check if right target zone entry occurred
				if not in_right_target and center[0] > right_target_x:
					right_target_entries += 1

				# check if fish is in left target
				if center[0] < left_target_x:
					#if not in_left_target:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('left_target','{:.2f}'.format(counter*spf)))
					left_target_frame_cnt += (frames_not_tracking + 1)
					in_left_target = True
					if not first_target_zone_entered:
						first_target_zone = 'left'
						first_target_zone_entered = True
				else:
					in_left_target = False

				# check if fish is in right target
				if center[0] > right_target_x:
					#if not in_right_target:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('right_target','{:.2f}'.format(counter*spf)))
					right_target_frame_cnt += (frames_not_tracking + 1)
					in_right_target = True
					if not first_target_zone_entered:
						first_target_zone = 'right'
						first_target_zone_entered = True
				else:
					in_right_target = False

				# check left target latency (if still applicable)
				if left_target_entries == 0:
					left_target_latency_frame_cnt += (frames_not_tracking + 1)

				# check right target latency (if still applicable)
				if right_target_entries == 0:
					right_target_latency_frame_cnt += (frames_not_tracking + 1)

				# check if in upper mirror zone
				if center[0] > upper_mirror_x1 and center[0] < upper_mirror_x2 and center[1] < upper_mirror_y:
					#if not in_upper_mirror:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('upper_mirror','{:.2f}'.format(counter*spf)))
					upper_mirror_frame_cnt += (frames_not_tracking + 1)
					in_upper_mirror = True
				else:
					in_upper_mirror = False

				# check if in lower mirror zone
				if center[0] > lower_mirror_x1 and center[0] < lower_mirror_x2 and center[1] > lower_mirror_y:
					#if not in_lower_mirror:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('lower_mirror','{:.2f}'.format(counter*spf)))
					lower_mirror_frame_cnt += (frames_not_tracking + 1)
					in_lower_mirror = True
				else:
					in_lower_mirror = False

				# check if in upper left thigmotaxis zone
				if center[0] > thigmo_ul_x1 and center[0] < thigmo_ul_x2 and center[1] < thigmo_upper_y:
					#if not in_ul_thigmo:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('ul_thigmo','{:.2f}'.format(counter*spf)))
					ul_thigmo_frame_cnt += (frames_not_tracking + 1)
					in_ul_thigmo = True
				else:
					in_ul_thigmo = False

				# check if in upper right thigmotaxis zone
				if center[0] > thigmo_ur_x1 and center[0] < thigmo_ur_x2 and center[1] < thigmo_upper_y:
					#if not in_ur_thigmo:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('ur_thigmo','{:.2f}'.format(counter*spf)))
					ur_thigmo_frame_cnt += (frames_not_tracking + 1)
					in_ur_thigmo = True
				else:
					in_ur_thigmo = False

				# check if in lower left thigmotaxis zone
				if center[0] > thigmo_ll_x1 and center[0] < thigmo_ll_x2 and center[1] > thigmo_lower_y:
					#if not in_ll_thigmo:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('ll_thigmo','{:.2f}'.format(counter*spf)))
					ll_thigmo_frame_cnt += (frames_not_tracking + 1)
					in_ll_thigmo = True
				else:
					in_ll_thigmo = False

				# check if in lower right thigmotaxis zone
				if center[0] > thigmo_lr_x1 and center[0] < thigmo_lr_x2 and center[1] > thigmo_lower_y:
					#if not in_ur_thigmo:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('ur_thigmo','{:.2f}'.format(counter*spf)))
					lr_thigmo_frame_cnt += (frames_not_tracking + 1)
					in_lr_thigmo = True
				else:
					in_lr_thigmo = False

				# check if fish is in corners (may use for thigmotaxis score)
				if center[0] < thigmo_ul_x1 and center[1] < thigmo_upper_y:
					#if not in_ul_corner:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('ul_corner','{:.2f}'.format(counter*spf)))
					ul_corner_frame_cnt += (frames_not_tracking + 1)
					in_ul_corner = True
				else:
					in_ul_corner = False

				if center[0] < thigmo_ll_x1 and center[1] > thigmo_lower_y:
					#if not in_ll_corner:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('ll_corner','{:.2f}'.format(counter*spf)))
					ll_corner_frame_cnt += (frames_not_tracking + 1)
					in_ll_corner = True
				else:
					in_ll_corner = False

				if center[0] > thigmo_ur_x2 and center[1] < thigmo_upper_y:
					#if not in_ur_corner:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('ur_corner','{:.2f}'.format(counter*spf)))
					ur_corner_frame_cnt += (frames_not_tracking + 1)
					in_ur_corner = True
				else:
					in_ur_corner = False

				if center[0] > thigmo_lr_x2 and center[1] > thigmo_lower_y:
					#if not in_lr_corner:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('lr_corner','{:.2f}'.format(counter*spf)))
					lr_corner_frame_cnt += (frames_not_tracking + 1)
					in_lr_corner = True
				else:
					in_lr_corner = False

				if not in_left_target and not in_right_target and not in_ll_corner and not in_lr_corner and not in_ul_corner and not in_ur_corner and not in_ll_thigmo and not in_lr_thigmo and not in_ul_thigmo and not in_ur_thigmo and not in_lower_mirror and not in_upper_mirror:
					#if not in_center:
					#	with open(zone_log, 'a') as f:
					#		writer = csv.writer(f)
					#		writer.writerow(('center','{:.2f}'.format(counter*spf)))
					in_center = True
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

				if not acquired:
					frames_b4_acq += 1

				if acquired:
					potential_freeze_frames += 1

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
				cv2.rectangle(frame,(0,0),(left_target_x,vidHeight),(0,255,0),2)

			if in_right_target:
				cv2.rectangle(frame,(right_target_x,0),(vidWidth,vidHeight),(0,255,0),2)

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

			if False:
				cv2.rectangle(frame,(0,0),(left_target_x,vidHeight),(0,255,0),2)
				cv2.rectangle(frame,(right_target_x,0),(vidWidth,vidHeight),(0,255,0),2)
				cv2.rectangle(frame,(upper_mirror_x1,0),(upper_mirror_x2,upper_mirror_y),(255,0,0),2)
				cv2.rectangle(frame,(lower_mirror_x1,lower_mirror_y),(lower_mirror_x2,vidHeight),(255,0,0),2)
				cv2.rectangle(frame,(thigmo_ul_x1,0),(thigmo_ul_x2,thigmo_upper_y),(0,0,255),2)
				cv2.rectangle(frame,(thigmo_ur_x1,0),(thigmo_ur_x2,thigmo_upper_y),(0,0,255),2)
				cv2.rectangle(frame,(thigmo_ll_x1,thigmo_lower_y),(thigmo_ll_x2,vidHeight),(0,0,255),2)
				cv2.rectangle(frame,(thigmo_lr_x1,thigmo_lower_y),(thigmo_lr_x2,vidHeight),(0,0,255),2)

			if acquired and not (in_left_target or in_right_target or in_ll_corner or in_lr_corner or in_ul_corner or \
								in_ur_corner or in_ll_thigmo or in_lr_thigmo or in_ul_thigmo or in_ur_thigmo or \
								in_lower_mirror or in_upper_mirror):
				cv2.rectangle(frame,(thigmo_ul_x1, thigmo_upper_y),(thigmo_lr_x2,thigmo_lower_y),(255,255,0),2)
			# show frame
			if show_images:
				#cv2.imshow('image',frame)
				cv2.imshow('image', cv2.resize(frame, (0,0), fx=0.5, fy=0.5))

			endOfLoop = time.time()

			#print 'time of loop: ' + str(round(time.time()-beginningOfLoop,4))

			k = cv2.waitKey(1)
			if k == 27:
				break

			if acquired:
				counter+=1

			faux_counter+=1

			if first_pass:
				first_pass = False

		print '#' * 45
		print 'Trial Metrics: '
		print '#' * 45
		print 'Activity level: ' + str(activity_level) + ' cm'

		if leftside_high or rightside_high:
			if (trained_high and leftside_high) or (trained_low and not leftside_high):
				reinforced_latency = left_target_latency_frame_cnt * spf
				time_in_reinforced_target = left_target_frame_cnt * spf
				num_entries_reinforced = left_target_entries
				if left_target_frame_cnt == 0:
					prop_time_reinforced = 0.0
				else:
					prop_time_reinforced = (float(left_target_frame_cnt) / counter) * 100.0
				if first_target_zone == 'left':
					first_target_zone = 'Reinforced'
				elif first_target_zone == 'right':
					first_target_zone = 'Non-Reinforced'
			elif (trained_high and rightside_high) or (trained_low and not rightside_high):
				reinforced_latency = right_target_latency_frame_cnt * spf
				time_in_reinforced_target = right_target_frame_cnt * spf
				num_entries_reinforced = right_target_entries
				if right_target_frame_cnt == 0:
					prop_time_reinforced = 0.0
				else:
					prop_time_reinforced = (float(right_target_frame_cnt) / counter) * 100.0
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
				non_reinforced_latency = left_target_latency_frame_cnt * spf
				time_in_non_reinforced_target = left_target_frame_cnt * spf
				num_entries_non_reinforced = left_target_entries
				if left_target_frame_cnt == 0:
					prop_time_non_reinforced = 0.0
				else:
					prop_time_non_reinforced = (float(left_target_frame_cnt) / counter) * 100.0
			elif (trained_low and rightside_high) or (trained_high and not rightside_high):
				non_reinforced_latency = right_target_latency_frame_cnt * spf
				time_in_non_reinforced_target = right_target_frame_cnt * spf
				num_entries_non_reinforced = right_target_entries
				if right_target_frame_cnt == 0:
					prop_time_non_reinforced = 0.0
				else:
					prop_time_non_reinforced = (float(right_target_frame_cnt) / counter) * 100.0
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
			prop_time_mirror = (float(upper_mirror_frame_cnt + lower_mirror_frame_cnt) / counter) * 100.0
		print 'Proportion time in Mirror Zones: ' + str(prop_time_mirror) + '%'

		# Thigmotaxis metrics
		thigmo_frame_cnt = ll_thigmo_frame_cnt + lr_thigmo_frame_cnt + ul_thigmo_frame_cnt + ur_thigmo_frame_cnt
		thigmotaxis_score = thigmo_frame_cnt * spf
		print 'Time in Thigmo Zones: ' + str(thigmotaxis_score) + ' secs'
		# TODO: is this correct for thigmotaxis score?

		if thigmo_frame_cnt == 0:
			prop_time_thigmo = 0.0
		else:
			prop_time_thigmo = (float(thigmo_frame_cnt) / counter) * 100.0
		print 'Proportion time in Thigmotaxis Zones: ' + str(prop_time_thigmo) + '%'

		if (100 - (prop_time_reinforced+prop_time_non_reinforced+prop_time_mirror+prop_time_thigmo)) == 0:
			prop_time_center = 0.0
		else:
			prop_time_center = 100 - (prop_time_reinforced+prop_time_non_reinforced+prop_time_mirror+prop_time_thigmo)
		print 'Proportion time in Center Zone: ' + str(prop_time_center) + '%'

		total_time = counter * spf
		time_in_center = total_time - (time_in_reinforced_target-time_in_non_reinforced_target-time_in_mirror-thigmotaxis_score)
		print 'Time in Center Zone: ' + str(time_in_center) + ' secs'

		time_in_corners = (ul_corner_frame_cnt + ur_corner_frame_cnt + ll_corner_frame_cnt + lr_corner_frame_cnt) * spf

		if (ul_corner_frame_cnt + ur_corner_frame_cnt + ll_corner_frame_cnt + lr_corner_frame_cnt) == 0:
			prop_corners = 0.0
		else:
			prop_corners = (float(ul_corner_frame_cnt + ur_corner_frame_cnt + ll_corner_frame_cnt + lr_corner_frame_cnt) / counter) * 100.0
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
				                'Prop.Center.%', 'Activity.Level.CM', \
				                'Time.Corners.Secs', 'Prop.Corners.%', \
				                'First.Target.Zone', \
				                'Left.Target.Secs', 'Right.Target.Secs', \
				                'UL.Thigmo.Secs', 'LL.Thigmo.Secs', \
				                'UR.Thigmo.Secs', 'LR.Thigmo.Secs', \
				                'Center.Secs', 'Upper.Mirror.Secs', \
				                'Lower.Mirror.Secs', 'Total.Secs'))

		# Open csv file in append mode
		with open(csv_filename, 'a') as f:
			writer = csv.writer(f)

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
							 '{:.2f}'.format((left_target_frame_cnt+right_target_frame_cnt+ul_thigmo_frame_cnt+ll_thigmo_frame_cnt+ur_thigmo_frame_cnt+lr_thigmo_frame_cnt+center_frame_cnt+lower_mirror_frame_cnt+upper_mirror_frame_cnt)*spf)))
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
				                'Lost.Track.Frame.Cnt', 'Cant.Decide', \
				                'Times.Lost.Track', 'Frames.Before.Acq', \
				                'Frames.Btwn.Tracks'))

		# Open csv file in append mode
		with open(tracker_log, 'a') as f:
			writer = csv.writer(f)

			# write the data (limit all decimals to 2 digits)
			writer.writerow((filename, counter,  '{:.2f}'.format(counter*spf), \
							lost_track_frame_cnt, cant_decide, times_lost_track, \
							frames_b4_acq, frames_since_last_track))
		print 'Done with ' + tracker_log + '!'
		print '#' * 45

		if False:
			print '\n'
			print '#'*45
			print 'Plot some stuff!'
			#fig = plt.figure()
			#ax = fig.add_subplot(111)

			#y = [prop_time_reinforced, prop_time_non_reinforced, prop_time_mirror, prop_time_thigmo, prop_time_center]
			#N = len(y)
			#x = range(N)
			#width = 1/6.0
			#ax.bar(x, y, width, color='blue')
			#ax.set_xticklabels(['_', \
			#					'Reinforced', \
			#                    'Non.Reinforced', \
			#                    'Mirror', \
			#                    'Thigmo', \
			#                    'Center'])
			#ax.set_ylabel('% in zone')
			#ax.set_title('Prop.In.Zones')
			#fig.savefig('Prop_In_Zones.png', bbox_inches='tight')

			print '#'*45, '\n'

		### after the program exits, print some useful stuff to the screen
		# first calculate realized fps
		print '#' * 45
		print 'Some Useful stuff...'
		print 'Frame count: ' + str(counter)
		print 'Total Video Time: ' + str(counter*spf) + ' secs'
		print 'Lost track frame count: ' + str(lost_track_frame_cnt)
		print 'Cant decide: ' + str(cant_decide)
		print 'Times lost track: ' + str(times_lost_track)
		print 'Frames before acquisition: ' + str(frames_b4_acq)
		print '\nThis program took ' + str(time.time() - start_time) + " seconds to run."
		print '#' * 45
		cv2.destroyAllWindows()

if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--path2video', help = 'Path to video file including the filename', required=True)
	ap.add_argument('-c', '--config_file', help = 'Path to config file (json) including the filename', required=True)
	ap.add_argument('-s', '--show_images', help='Whether or not show images (slows processing)',action='store_true')
	args = vars(ap.parse_args())

	path = args['path2video']

	#path = r'D:\num\gambusia_17_201_male_winston_9_1_7_14_50_none_R.mp4'
	#path = r'D:\num\gambusia_17_376_female_glinda_9_1_7_14_50_none_L.mp4'
	#path = r'D:\new_num\gambusia_17_355_female_winnie_9_1_9_12_75_none_R.mp4'
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

	if args['show_images']:
		show_images = True
	else:
		show_images = False

	this_tracker = Tracker(vid_file, config_file, show_images)
	this_tracker.run_numerosity_tracker()
