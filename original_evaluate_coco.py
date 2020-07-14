from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print ('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
annFile = '/home/mayank_s/codebase/cplus_plus/ai/darknet_AlexeyAB/coco/annotations/instances_val2014.json'
cocoGt=COCO(annFile)

#initialize COCO detections api
# resFile='data/instances_val2014_fakebbox100_results.json'
# resFile='/home/mayank_s/codebase/others/yolo/yolov4/yolov3/results.json'
resFile='/home/mayank_s/codebase/others/myrepo/mAP/scripts/extra/results.json'
# resFile='scr/result_yolov4.json'


cocoDt=cocoGt.loadRes(resFile)

# imgIds = [42, 73, 74, 133, 136, 139, 143, 164, 192, 196, 208, 241, 257, 283, 285, 294, 328, 338, 357, 359, 360, 387, 395, 397, 400, 415, 428, 459, 472, 474, 486, 488, 502, 520, 536, 544, 564, 569, 589, 590, 599, 623, 626, 632, 636, 641, 661, 675, 692, 693, 699, 711, 715, 724, 730, 757, 761, 764, 772, 775, 776, 785, 802, 810, 827, 831, 836, 872, 873, 885, 923, 939, 962, 969, 974, 985, 987, 999, 1000, 1029, 1064, 1083, 1089, 1103, 1138, 1146, 1149, 1153, 1164, 1171, 1176, 1180, 1205, 1228, 1244, 1268, 1270, 1290, 1292]
####################33333
# if dataloader is None:
# this is modified to work only for coco validation result for other you need to make modificationif imgIds list
imgIds=[]
with open('scripts/extra/coco_5k_img_list.txt', 'r') as f:
    imgIds = [int(line.strip()) for line in f]
#########################
# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
# cocoEval.params.areaRng = cocoEval.params.areaRng[:1]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
