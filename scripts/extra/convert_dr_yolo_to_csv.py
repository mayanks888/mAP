import os
import re
import pandas as pd
#you only need to provide result.txt from yolo output



# with open("coco_paper_names.txt") as f:
# with open("coco_class_list.txt") as f:
with open("bdd_names_list.txt") as f:
  obj_list = f.readlines()
## remove whitespace characters like `\n` at the end of each line
  obj_list = [x.strip() for x in obj_list]


# IN_FILE = '/home/mayank_s/codebase/cplus_plus/ai/darknet_AlexeyAB/mank_result/result_yolo_v4.txt'
# IN_FILE = '/home/mayank_s/codebase/cplus_plus/ai/darknet_AlexeyAB/mank_result/result_yolo_v3.txt'
IN_FILE = '/home/mayank_s/codebase/cplus_plus/ai/darknet_AlexeyAB/mank_result/result_gaussian_bdd.txt'

SEPARATOR_KEY = 'Enter Image Path:'
IMG_FORMAT = '.jpg'
bblabel=[]
outfile = None
flag=False
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
            #############################3333

            file_name=image_name.split('COCO_val2014_')[-1]
            # file_name="00100200"
            # b = [(lambda x: x.strip('0') if isinstance(x, str) and len(x) != 1 else x)(x) for x in file_name]
            #
            # trailing_removed = [s.rstrip("0") for s in file_name]
            # leading_removed = [s.lstrip("0") for s in file_name]
            # both_removed = [s.strip("0") for s in file_name]
            file_name=file_name.lstrip("0")
            ####################################
            # file_name=image_name
            flag=True

        elif flag:
            # split line on first occurrence of the character ':' and '%'
            class_name, info = line.split(':', 1)
            # if class_name.split(" "):
            # if len(class_name.split(" "))>1:
            #     print(class_name)
            #     class_name=class_name.replace(" ", "_")
            #     print(class_name)
            confidence, bbox = info.split('%', 1)
            # get all the coordinates of the bounding box
            bbox = bbox.replace(')','') # remove the character ')'
            # go through each of the parts of the string and check if it is a digit
            left, top, width, height = [int(s) for s in bbox.split() if s.lstrip('-').isdigit()]
            right = left + width
            bottom = top + height
            # outfile.write("{} {} {} {} {} {}\n".format(class_name, float(confidence)/100, left, top, right, bottom))
            conf=float(confidence)/100
            obj_name=class_name
            print(obj_name)
            obj_id=obj_list.index(obj_name)
            # obj_id = obj['category_id']
            img_width = img_height = 0
            data_label = [file_name, img_width, img_height, obj_name, obj_id, left, top, right, bottom, conf]
            bblabel.append(data_label)


columns = ['filename', 'width', 'height', 'class', 'obj_category', 'xmin', 'ymin', 'xmax', 'ymax', 'conf']

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('yolo_txt_to_csv.csv', index=False)
print("Conversion completed!")
