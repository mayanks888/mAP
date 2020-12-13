from collections import namedtuple
import os
import pandas as pd
# csv_path='/media/mayank_sati/DATA/datasets/traffic_light/seol-tl/seol_valid.csv'
# csv_path='/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/evaluation/2019-09-27-14-39-41_test_scaled_eval_farm.csv'
csv_path='/home/mayank_s/codebase/others/centernet/mayank/CenterNet/src/centernet_prediction_val.csv'
# data = pd.read_csv('/media/mayank_sati/DATA/datasets/traffic_light/BDD/csvfiles/BBD_daytime_val.csv')
data = pd.read_csv(csv_path)
# image_path='/home/mayank_s/datasets/bdd/bdd100k_images/bdd100k/images/100k/val'
image_path=''
print(data.head())

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    # filename='img_name'
    # data = namedtuple('data', ['img_name', 'obj_class'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


grouped = split(data, 'filename')

# save_path="/home/mayank_s/datasets/detection_result/BDD_groundtruth"
save_path="/home/mayank_s/Desktop/bdd/bdd_centernet_predict"
counter=0
for group in grouped:

        # filename = group.filename.encode('utf8')
        filename = group.filename
        check_image_path=image_path+"/"+filename
        if os.path.exists(check_image_path):
            counter+=1
            print(counter)
        # txt_file_name=filename.split('.')[0]
            txt_file_name=filename.split('.jpg')[0]
            # txt_file_name=filename.split('/')[-1]

            txt_file_name=txt_file_name+".txt"
            txt_file_path=save_path+"/"+txt_file_name
            with open(txt_file_path, 'w') as f:
                for index, row in group.object.iterrows():
                    xmin=row['xmin']
                    xmax=row['xmax']
                    ymin=row['ymin']
                    ymax=row['ymax']
                    class_name=row['class']
                    # score=row['score']
                    if len(class_name.split(" ")) > 1:
                        print(class_name)
                        class_name = class_name.replace(" ", "_")
                        print(class_name)

                    # f.write(str(bboxcls) + " " + " ".join([str(a) for a in bb]) + '\n')
                    bb = ((xmin), (ymin), (xmax), (ymax))
                    # f.write(str(class_name) + " " + " ".join([str(a) for a in bb]) + '\n')

        else:
            print(filename)
