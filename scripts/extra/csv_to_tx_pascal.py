#never user the json file from the coco result, they something like (80class to 90 object thus destroy the whole thing somehow)
# so try to create you own csv from prediction and then create txt file to compar properly
from collections import namedtuple
import os
import pandas as pd

csv_path='yolo1.csv'
#where to place
save_path = "/home/mayank_s/datasets/detection_result/fake2"

data = pd.read_csv(csv_path)
print(data.head())

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    # filename='img_name'
    # data = namedtuple('data', ['img_name', 'obj_class'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


grouped = split(data, 'filename')

if not os.path.exists(save_path):
  os.makedirs(save_path)
counter=0
for group in grouped:

        # filename = group.filename.encode('utf8')
            filename = group.filename
            # @@@@@@@@@@@@@
            # this is only specific to coco 2014 validation
            test_string = 'COCO_val2014_'
            K = '0'
            # No. of zeros required
            N = 12-len(str(filename))
            # using format()
            # Append K character N times
            temp = '{:<' + K + str(len(test_string) + N) + '}'
            res = temp.format(test_string)
            # @@@@@@@@@@@@@@@@@
            txt_file_name=res+str(filename)+".txt"
            # txt_file_name=txt_file_name+".txt"
            txt_file_path=save_path+"/"+txt_file_name
            with open(txt_file_path, 'w') as f:
                for index, row in group.object.iterrows():
                    xmin=int(row['xmin'])
                    xmax=int (row['xmax'])
                    ymin=int(row['ymin'])
                    ymax=int(row['ymax'])
                    score=row['conf']
                    obj_name=row["class"]
                    if len(obj_name.split(" ")) > 1:
                        print(obj_name)
                        obj_name = obj_name.replace(" ", "_")
                        print(obj_name)
                    # f.write(str(bboxcls) + " " + " ".join([str(a) for a in bb]) + '\n')
                    bb = ((score),(xmin), (ymin), (xmax), (ymax))
                    f.write(str(obj_name) + " " + " ".join([str(a) for a in bb]) + '\n')
        #
        # else:
        #     print(filename)
