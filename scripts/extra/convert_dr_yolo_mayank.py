import os
import re
#you only need to provide result.txt from yolo output

# make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# change directory to the one with the files to be changed
# parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
# DR_PATH = os.path.join(parent_path, 'input','detection-results')
# IN_FILE = 'result2.txt'
#result from yolo file
# IN_FILE = '/home/mayank_s/codebase/cplus_plus/ai/darknet_AlexeyAB/yolo_v4_coco_412.txt'
IN_FILE = '/home/mayank_s/codebase/cplus_plus/ai/darknet_AlexeyAB/result_yolo_512_v3.txt'

#where to place
# DR_PATH = "/home/mayank_s/datasets/detection_result/yolo_v4_coco_14_412_size"
DR_PATH = "/home/mayank_s/datasets/detection_result/yolo_v3_coco_516"

if not os.path.exists(DR_PATH):
  os.makedirs(DR_PATH)
#print(DR_PATH)

os.chdir(DR_PATH)

SEPARATOR_KEY = 'Enter Image Path:'
IMG_FORMAT = '.jpg'

outfile = None
with open(IN_FILE) as infile:
    for line in infile:
        if SEPARATOR_KEY in line:
            if IMG_FORMAT not in line:
                break
            # get text between two substrings (SEPARATOR_KEY and IMG_FORMAT)
            image_path = re.search(SEPARATOR_KEY + '(.*)' + IMG_FORMAT, line)
            # get the image name (the final component of a image_path)
            # e.g., from 'data/horses_1' to 'horses_1'
            image_name = os.path.basename(image_path.group(1))
            # close the previous file
            if outfile is not None:
                outfile.close()
            # open a new file
            outfile = open(os.path.join(DR_PATH, image_name + '.txt'), 'w')
        elif outfile is not None:
            # split line on first occurrence of the character ':' and '%'
            class_name, info = line.split(':', 1)
            # if class_name.split(" "):
            if len(class_name.split(" "))>1:
                print(class_name)
                class_name=class_name.replace(" ", "_")
                print(class_name)
            confidence, bbox = info.split('%', 1)
            # get all the coordinates of the bounding box
            bbox = bbox.replace(')','') # remove the character ')'
            # go through each of the parts of the string and check if it is a digit
            left, top, width, height = [int(s) for s in bbox.split() if s.lstrip('-').isdigit()]
            right = left + width
            bottom = top + height
            outfile.write("{} {} {} {} {} {}\n".format(class_name, float(confidence)/100, left, top, right, bottom))
