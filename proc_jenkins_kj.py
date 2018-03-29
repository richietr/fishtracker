import os
import json
import types
import sys
from shutil import copyfile
from numerosityTracker_NEW import Tracker

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

################################################################################
config_file = os.path.join(os.getcwd(),'numerosity_tracker_config_NEW.json')
fish_json = os.path.join(os.getcwd(),'fish.json')
grid_json = os.path.join(os.getcwd(),'numerosity_grid.json')
show_images = False

proc_jenkins_config_json = parse_json("proc_jenkins_config.json")

share_base_loc = '/home/charliebrown/KevinJames/jenkinsData/numerosity/'
#share_base_loc = '/home/charliebrown/LarryFitzgerald/jenkinsData/'

results_file = share_base_loc + 'numerosity_log.csv'
local_results_file = 'numerosity_log.csv'

rounds_to_proc = []
rounds_to_exclude = []
fish_to_exclude = []
pies_to_process = []
vid_proc_list = []
last_line = None

# TODO need to work on appending results
# Delete any old results
#if os.path.exists(local_results_file):
#   os.remove(local_results_file)

#print os.listdir(share_base_loc)

# list all rounds in location
for name in os.listdir(share_base_loc):
    if os.path.isdir(os.path.join(share_base_loc, name)):
        rounds_to_proc.append(name)

# check for any rounds to exclude
#print rounds_to_proc
#print proc_jenkins_config_json
if proc_jenkins_config_json is not None:
    if "exclude_rounds" in proc_jenkins_config_json:
        rounds_to_exclude = proc_jenkins_config_json["exclude_rounds"]
#print rounds_to_exclude

for tmp in rounds_to_exclude:
    tmp = str(tmp)
    #print tmp
    if tmp in rounds_to_proc:
        rounds_to_proc.remove(tmp)

#print rounds_to_proc

# check for fish to exclude
if "exclude_fish" in proc_jenkins_config_json:
    fish_to_exclude = proc_jenkins_config_json["exclude_fish"]

#print fish_to_exclude

# open file listing all processed videos
if os.path.exists(os.path.join(share_base_loc, 'vid_proc_list.txt')):
    with open(os.path.join(share_base_loc, 'vid_proc_list.txt')) as f:
        vid_proc_list = f.readlines()
        vid_proc_list = [x.strip() for x in vid_proc_list]

    print vid_proc_list

for round in rounds_to_proc:
    print 'Processing round: ' + str(round)

    # list all pies for this round
    pies_to_process = []
    for pie in os.listdir(os.path.join(share_base_loc, round)):
        if os.path.isdir(os.path.join(share_base_loc, round, pie)):
            print 'Processing pie: ' + pie
            # list all videos for this pie
            for vid in os.listdir(os.path.join(share_base_loc, round, pie)):
                skip = False
                print 'Checking ' + vid

                if vid.endswith('.mp4'):
                    for fish in fish_to_exclude:
                        if fish in vid:
                            print "Not processing because fish is in exclusion list: " + vid
                            skip = True
                    if vid in vid_proc_list:
                        print "Not processing because video has already been processed: " + vid
                        skip = True

                    # Parse out filepath
                    path_strip = os.path.splitext(vid)[0]
                    #TODO make work for windows and linux
                    path_parts = path_strip.split('/')
                    #path_parts = path_strip.split('\\')
                    filename = path_parts[len(path_parts)-1]
                    filename_parts = filename.split('_')

                    if len(filename_parts) is 5:
                        print "Not processing because video is a record only video: " + vid
                        skip = True
                    else:
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

                        if int(day) < 9 or fedside != 'none':
                            print "Not processing because video is not a testing video: " + vid
                            skip = True

                    if skip is False:
                        print "Processing " + vid
                        with open("envvar.txt", "w") as f:
                            f.write("VIDEO_FILE=" + vid)
                        vid_file = os.path.join(share_base_loc, round, pie, vid)
                        this_tracker = Tracker(vid_file, config_file, fish_json, grid_json, show_images)
                        this_tracker.run_numerosity_tracker()

                        # Manage results
                        if os.path.exists(results_file):
                            # just want to append last line of local file to end of remote file
                            with open(local_results_file, "r") as f:
                                last_line = f.readlines()[-1]
                            if last_line is not None:
                                with open(results_file, "a") as f:
                                    f.write(last_line + '\n')
                            else:
                                print "Error> Unable to write results, exiting"
                                sys.exit(1)
                        else:
                            # since a results file doesn't exist just copy the local one
                            copyfile(local_results_file, results_file)

                        # append vid file to processed videos file
                        with open(os.path.join(share_base_loc, 'vid_proc_list.txt'), 'a') as f:
                            f.write(vid + '\n')

                        # only processing one at a time so exit
                        sys.exit(0)
