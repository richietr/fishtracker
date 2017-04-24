import numpy as np
import cv2
import sys
import time
import math
import os
import csv
import matplotlib.pyplot as plt
import json
import types
import argparse

def parseJSON():
	with open(os.path.join('config.json'), 'r') as jsonFile:
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
		print 'Error> Config dictionary is None, problem parsing config.json'
	
	return config_dict

def true_distance(a, b):
	d = math.sqrt((float(a[0]) - b[0])**2 + (float(a[1]) - b[1])**2)
	return d

# crops based on boundaries
def apply_mask(frame, lower_bound, upper_bound, left_bound, right_bound):

	# initialize mask to all 0s
	mask = np.zeros((vidHeight, vidWidth, 3),np.uint8)

	# use rectangle bounds for masking
	mask[lower_bound:upper_bound,left_bound:right_bound] = frame[lower_bound:upper_bound,left_bound:right_bound]

	return mask

def apply_special_mask(frame, corner_x1, corner_x2, corner_y1, corner_y2, target_x1, target_x2, corner_y3, corner_y4):

	# initialize mask to all 0s
	mask = np.zeros((vidHeight, vidWidth, 3),np.uint8)

	# use rectangle bounds for masking
	mask[corner_y1:corner_y2,corner_x1:corner_x2] = frame[corner_y1:corner_y2,corner_x1:corner_x2]
	mask[corner_y2:corner_y3,target_x1:target_x2] = frame[corner_y2:corner_y3,target_x1:target_x2]
	mask[corner_y3:corner_y4,corner_x1:corner_x2] = frame[corner_y3:corner_y4,corner_x1:corner_x2]
	
	return mask

# blurs a frame and crops based on boundaries
def blur_and_mask(frame, lower_bound, upper_bound, left_bound, right_bound):
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

def mergeContours(frame):
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


# returns centroid from largest contour from a binary image
def returnLargeContour(frame,totalVideoPixels, unified):

	global cant_decide

	# find all contours in the frame
	contours = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#print "number of contours: " + str(len(contours[0])) + "\n"
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
			if area < 1000.0 and area > 100.0 and max(h,w) < 75 and max(h,w) > 5:
				idx = area_list.index(area)
				potential_tracks.append(contours[idx])

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
def getBackgroundImage(vid,numFrames,length):

	print '#' * 45
	print 'Getting background image...'

	frames2skip = NUM_FRAMES_TO_SKIP*1.0
	frames2trim = NUM_FRAMES_TO_TRIM*1.0

	# For now skipping some frames at beginning of sociality where the pipe is 
	# being removed and water takes a little while to settle
	j = 0
	while j < frames2skip:
		vid.read()
		j+=1
	length = length - frames2skip - frames2trim

	# set a counter
	i = 0
	#vid.set(1, 200) # TODO: what is this doing?
	_,frame = vid.read()
	frameCnt = 1

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
		if i < numFrames-1:
			for j in range(1,skip):
				vid.read()
				frameCnt += 1

		# main function
		cv2.accumulateWeighted(frame,update,0.001)
		final = cv2.convertScaleAbs(update)

		# increment the counter
		i += 1

		# print something every 100 frames so the user knows the gears are grinding
		if i%100 == 0:
			print "Detecting background -- on frame " + str(frameCnt-skip) + " of " + str(length)

	print 'Background detection complete!'
	print '#' * 45

	return final


###############################################################################
#
#  TODO items
# - tracking multiple targets
# - handle shadows
# - code cleanup (think about how could be easily configured for sociality)
#
###############################################################################

if __name__ == '__main__':

	print time.strftime('%X %x %Z')
	start_time = time.time()

	global NUM_FRAMES_TO_SKIP
	global NUM_FRAMES_TO_TRIM
	global cant_decide
	cant_decide = 0

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--path2video", help = "Path to video file including the filename", required=True)
	ap.add_argument("-s", "--show_images", help="Whether or not show images (slows processing)",action="store_true")
	args = vars(ap.parse_args())	

	path = args["path2video"]
	path = 'numerosity_post_SOC_hachi_firsthalf_L_october_20_2016.wmv'
	if args["show_images"]:
		show_images = True
	else:
		show_images = False
	
	# Parse out filepath
	path_strip = os.path.splitext(path)[0]
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
	proportion = '0'#filename_parts[9]
	fedside = '0'#filename_parts[10]
	correctside = '0'#filename_parts[11]

	# defaults	
	SECS_B4_LOST = 1
	NUM_FRAMES_FOR_BACKGROUND = 1000
	EDGE_BUFFER = 25 #pixels
	CROP_X1 = 157#0
	CROP_X2 = 1059#1280
	CROP_Y1 = 194#0
	CROP_Y2 = 546#720
	TANK_LENGTH_CM = 40.64
	TANK_WIDTH_CM = 21.59
	TANK_UPPER_LEFT_X = 142
	TANK_UPPER_LEFT_Y = 180
	TANK_LOWER_LEFT_X = 157
	TANK_LOWER_LEFT_Y = 562
	TANK_UPPER_RIGHT_X = 1115
	TANK_UPPER_RIGHT_Y = 194
	TANK_LOWER_RIGHT_X = 1059
	TANK_LOWER_RIGHT_Y = 546
	TRACKING_WINDOW_LEN = 100 # TODO: Use cm instead of pixels
	TARGET_ZONE_CM = 10
	FREEZE_TIME_MIN_SECS = 3 # freeze must be 3 secs
	FREEZE_WINDOW_LEN = 40 # TODO: Use cm instead of pixels
	
	# Check if defaults overriden by config.json
	
	
	# data to pull out


	# Data will be stored in a csv file
	csv_filename = 'sociality_log.csv'
	freeze_log = 'soc_freeze_log.csv'
	tracker_log = 'soc_tracker_log.csv'

	counter = 0
	freeze_start = None
	freeze_zone = None
	freeze_event_cnt = 0
	freeze_frame_cnt = 0
	potential_freeze_frames = 0
	freeze_time_secs = 0	
	activity_level = 0
	
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

	# length of a pixel (assuming square)
	pixel_cm = TANK_LENGTH_CM/(TANK_UPPER_RIGHT_X-TANK_UPPER_LEFT_X)

	# cropping window
	upper_bound, left_bound, right_bound, lower_bound = CROP_Y2, CROP_X1, CROP_X2, CROP_Y1
	new_upper_bound, new_left_bound, new_right_bound, new_lower_bound = CROP_Y2, CROP_X1, CROP_X2, CROP_Y1

	# open the video
	cap = cv2.VideoCapture(path)
	if not cap.isOpened():
		print "ERROR> Could not open :",path

	# get some info about the video
	#length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	length = 6000
	print 'Number of frames: ' +  str(length)
	vidWidth  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	print 'Width: ' +  str(vidWidth)
	vidHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	print 'Height: ' +  str(vidHeight)
	fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	print 'FPS: ' +  str(fps)

	# seconds per frame
	spf = 1.0/fps

	NUM_FRAMES_TO_SKIP = 8 * fps
	NUM_FRAMES_TO_TRIM = 0

	# grab the 20th frame for drawing the rectangle
	i = 0
	while i < 20:
		ret,frame = cap.read()
		i += 1
	print "grabbed first frame? " + str(ret)

	# calculate background image of tank for x frames
	background = getBackgroundImage(cap,NUM_FRAMES_FOR_BACKGROUND,length)

	# blur and crop background and save a copy of the background image for reference
	bm_initial = blur_and_mask(background, lower_bound, upper_bound, left_bound, right_bound)
	cv2.imwrite("background.jpg",background)

	startOfTrial = time.time()
	cap = cv2.VideoCapture(path)

	if not cap.isOpened():
		print "ERROR> Could not open :",path
		sys.exit(1)

	first_pass = True
	prev_center = None

	print '\n\nProcessing...\n'
	while(cap.isOpened()):
	#while(counter < (length-2000)):

		#print "frame " + str(counter) + "\n\n"
		if not counter % 100:
			print 'Processing frame ' + str(counter) + '...'

		# for timing, maintaining constant fps
		beginningOfLoop = time.time()

		# skipping some frames at the beginning
		if first_pass:
			for i in range(0, int(fps)):
				cap.read()
		ret,frame = cap.read()

		if ret == False:
			print "didn't read frame from video file"
			break

		# blur and crop frame
		bm = blur_and_mask(frame, lower_bound, upper_bound, left_bound, right_bound)

		# find difference between frame and background
		difference = cv2.absdiff(bm, bm_initial)
		difference = apply_mask(difference, new_lower_bound, new_upper_bound, new_left_bound, new_right_bound)

		# find the centroid of the largest blob
		imgray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray,15,255,0)
		if show_images:
			cv2.imshow('thresh',thresh)

		if counter < NUM_FRAMES_TO_SKIP:
			unified = None
		else:
			unified = mergeContours(frame)
		#unified = None
		center = returnLargeContour(thresh, vidWidth*vidHeight, unified)
		#print "Center: " + str(center) + "\n"

		# calc distance between current center and previous center
		if not first_pass:
			if center is not None and prev_center is not None:
				# distance between points in pixels times cm in a single pixel
				activity_level += (true_distance(prev_center, center))*pixel_cm

		if center is not None:			

			prev_center = center
			
			if not acquired:
				acquired = True
			
			if not tracking:
				tracking = True
				print 'Target acquired... tracking...'
				print 'Frames since last tracking: ' + str(frames_not_tracking)
				if acquired:
					frames_since_last_track.append(frames_not_tracking)
				frames_not_tracking = 0
				
			
			#if counter > NUM_FRAMES_TO_SKIP:
			# shrink search window
			#new_upper_bound = center[1] + TRACKING_WINDOW_LEN
			#new_lower_bound = center[1] - TRACKING_WINDOW_LEN
			#if center[0] - TRACKING_WINDOW_LEN < TANK_UPPER_LEFT_X + EDGE_BUFFER:
			#	new_left_bound  = TANK_UPPER_LEFT_X + EDGE_BUFFER
			#else:				
			#	new_left_bound  = center[0] - TRACKING_WINDOW_LEN
			#if center[0] + TRACKING_WINDOW_LEN > TANK_UPPER_RIGHT_X - EDGE_BUFFER:
			#	new_right_bound = TANK_UPPER_RIGHT_X - EDGE_BUFFER
			#else:				
			#	new_right_bound = center[0] + TRACKING_WINDOW_LEN
			#print 'new_lower_bound=' + str(new_lower_bound)
			#print 'new_upper_bound=' + str(new_upper_bound)
			#print 'new_left_bound='  + str(new_left_bound)
			#print 'new_right_bound=' + str(new_right_bound)

			# check for freezing
			if freeze_start is None:
				freeze_start = center
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
					freeze_zone = 'center'

			# draw red circle on largest
			cv2.circle(frame,center,4,[0,0,255],-1)

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
			
			# assume fish is in same zone if not found
				
			if acquired:
				potential_freeze_frames += 1
			
		# draw box around track window
		cv2.rectangle(frame,(new_left_bound, new_lower_bound), (new_right_bound, new_upper_bound), (255,255,255),2)
			
		# draw green circle around freeze center
		if freeze_start is not None:
			cv2.circle(frame,freeze_start,FREEZE_WINDOW_LEN,[0,255,0],2)
				
		# show frame
		if show_images:
			cv2.imshow('image',frame)

		endOfLoop = time.time()

		#print "time of loop: " + str(round(time.time()-beginningOfLoop,4))

		k = cv2.waitKey(1)
		if k == 27:
			break

		counter+=1

		if first_pass:
			first_pass = False

	print '#' * 45
	print 'Trial Metrics: '
	print '#' * 45
	print 'Activity level: ' + str(activity_level) + ' cm'


	print 'Done Processing!'
	print '#' * 45
	print "\n\n"

	#print '#' * 45
	#print 'Write data out to ' + csv_filename

	# Check if csv file exists
	#if not os.path.isfile(csv_filename):

		# Open csv file in write mode
	#	with open(csv_filename, 'w') as f:
	#		writer = csv.writer(f)

			# write the header
	#		writer.writerow(('Fish.ID', 'Round', 'Day', \
	#		                'Session', 'Stimulus', 'Other.Stimulus', \
	#		                'Proportion', 'Fed.Side','Correct.Side', 'Trial.Type', \
	#		                'Reinforced.Latency.Secs', 'Non.Reinforced.Latency.Secs', \
	#		                'Time.Reinforced.Secs', 'Time.Non.Reinforced.Secs', \
	#		                'Reinforced.Entries', 'Non.Reinforced.Entries', \
	#		                'Prop.Reinforced.%', 'Prop.Non.Reinforced.%', \
	#		                'Time.Mirror.Secs', 'Prop.Mirror.%', \
	#		                'Time.Thigmo.Secs', 'Prop.Thigmo.%', \
	#		                'Prop.Center.%', 'Activity.Level.CM', \
	#		                'Time.Corners.Secs', 'Prop.Corners.%', \
	#		                'First.Target.Zone'))

	# Open csv file in append mode
	#with open(csv_filename, 'a') as f:
	#	writer = csv.writer(f)

		# write the data (limit all decimals to 2 digits)
		#writer.writerow((fishid, round_num, day, session, stimulus, that_stimulus, \
		#				 proportion, fedside, correctside, trial_type, \
		#				 reinforced_latency, non_reinforced_latency, \
		#				 '{:.2f}'.format(time_in_reinforced_target), '{:.2f}'.format(time_in_non_reinforced_target), \
		#				 num_entries_reinforced, num_entries_non_reinforced, \
		#				 '{:.2f}'.format(prop_time_reinforced), '{:.2f}'.format(prop_time_non_reinforced), \
		#				 '{:.2f}'.format(time_in_mirror), '{:.2f}'.format(prop_time_mirror), \
		#				 '{:.2f}'.format(thigmotaxis_score), '{:.2f}'.format(prop_time_thigmo), \
		#				 '{:.2f}'.format(prop_time_center), '{:.2f}'.format(activity_level), \
		#				 '{:.2f}'.format(time_in_corners), '{:.2f}'.format(prop_corners),
		#				 first_target_zone))
	#print 'Done with ' + csv_filename + '!'
	#print '#' * 45

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


	#print '\n'
	#print '#'*45
	#print 'Plot some stuff!'
	#fig = plt.figure()
	#ax = fig.add_subplot(111)

	#y = [prop_time_reinforced, prop_time_non_reinforced, prop_time_mirror, prop_time_thigmo, prop_time_center]
	#N = len(y)
	#x = range(N)
	#width = 1/6.0
	#ax.bar(x, y, width, color="blue")
	#ax.set_xticklabels(['_', \
	#					'Reinforced', \
	#                    'Non.Reinforced', \
	#                    'Mirror', \
	#                    'Thigmo', \
	#                    'Center'])
	#ax.set_ylabel('% in zone')
	#ax.set_title('Prop.In.Zones')
	#fig.savefig('Prop_In_Zones.png', bbox_inches='tight')

	#print '#'*45, '\n'

	### after the program exits, print some useful stuff to the screen
	# first calculate realized fps
	print '#' * 45
	print "Some Useful stuff..."
	print "Frame count: " + str(counter)
	print "Total Video Time: " + str(counter*spf) + " secs"
	print "Lost track frame count: " + str(lost_track_frame_cnt)
	print 'Cant decide: ' + str(cant_decide)
	print 'Times lost track: ' + str(times_lost_track)
	print 'Frames before acquisition: ' + str(frames_b4_acq)
	print "\nThis program took " + str(time.time() - start_time) + " seconds to run."
	print '#' * 45
	cv2.destroyAllWindows()
