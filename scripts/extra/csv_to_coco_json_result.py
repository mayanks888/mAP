from collections import namedtuple
import os
import pandas as pd
import json
from utils import *
coco91class = coco80_to_coco91_class()
csv_path='yolo_txt_to_csv.csv'
# csv_path='yolo1.csv'
data = pd.read_csv(csv_path)
print(data.head())

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    # filename='img_name'
    # data = namedtuple('data', ['img_name', 'obj_class'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

grouped = split(data, 'filename')

jdict= []
for group in grouped:

        # filename = group.filename.encode('utf8')
        filename = group.filename
        print(filename)
        for index, row in group.object.iterrows():
            xmin=(row['xmin'])
            ymin = (row['ymin'])
            width= (row['xmax'])-xmin
            height=(row['ymax'])-ymin
            # box_=[xmin,ymin,xmax,ymax]
            # box2=xyxy2xywh(box_)
            # obj_id = obj['category_id']
            # print(obj_id)
            score=row['conf']
            obj_name=row["class"]
            obj_cat=row["obj_category"]
            ################3

            obj_cat=coco91class[int(obj_cat)]
            #####################
            bbox = ((xmin), (ymin), (width), (height))
            # bbox = box2
            jdict.append({'image_id': int(filename), 'category_id': obj_cat, 'bbox': [round(x, 3) for x in bbox], 'score': round(score, 5)})

print('\nGenerating json detection for pycocotools...')
with open('results.json', 'w') as file:
    json.dump(jdict, file)
