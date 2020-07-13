
import numpy as np
#
# def xyxy2xywh(x):
#     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
#     y =  np.zeros_like(x)
#     x=np.array(x)
#     y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
#     y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
#     y[:, 2] = x[:, 2] - x[:, 0]  # width
#     y[:, 3] = x[:, 3] - x[:, 1]  # height
#     return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y =  np.zeros_like(x)
    x=np.array(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y

# def xywh2xyxy(x):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y =  np.zeros_like(x)
#     x = np.array(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y =  np.zeros_like(x)
    x = np.array(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[:1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[:1] + x[3] / 2  # bottom right y
    return y

# def xyxy2xywh(bbox):
#     _bbox = bbox.tolist()
#     return [
#         _bbox[0],
#         _bbox[1],
#         _bbox[2] - _bbox[0] + 1,
#         _bbox[3] - _bbox[1] + 1,
#     ]
