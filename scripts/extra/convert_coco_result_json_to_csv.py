import sys
import os
import glob
import json
import pandas as pd


with open("coco_class_list.txt") as f:
  obj_list = f.readlines()
## remove whitespace characters like `\n` at the end of each line
  obj_list = [x.strip() for x in obj_list]


filepath="/home/mayank_s/codebase/others/yolo/yolov4/yolov3/results.json"
# create VOC format files
bblabel=[]
json_list = glob.glob(filepath)
if len(json_list) == 0:
  print("Error: no .json files found in detection-results")
  sys.exit()
for tmp_file in json_list:
  #print(tmp_file)
  # 1. create new file (VOC format)
  # with open(tmp_file.replace(".json", ".txt"), "a") as new_f:
    data = json.load(open(tmp_file))
    for obj in data:
      # image_id; ": 54959, "; category_id; ": 78, "; bbox; ": [433.644, 76.958, 199.733, 113.05], "; score; ": 0.98185},
      file_name = obj['image_id']
      conf = obj['score']
      xmin = obj['bbox'][0]
      ymin = obj['bbox'][1]
      xmax = obj['bbox'][2]+xmin
      ymax = obj['bbox'][3]+ymin
      obj_id = obj['category_id'][0]
      print(obj_id)
      obj_name = obj_list[int(obj_id)]
      print(obj_name)
      width=height=0
      data_label = [file_name, width, height, obj_name, xmin, ymin, xmax, ymax,conf]
      # data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax]
      if not ((xmin == xmax) and (ymin == ymax)):
        bblabel.append(data_label)
        # print(file_name)
        # print()
      else:
        print("file_name")

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax','conf']

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('yolo1.csv',index=False)

print("Conversion completed!")
