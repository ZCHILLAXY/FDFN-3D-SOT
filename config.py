#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import os.path as osp
from easydict import EasyDict as edict
import math


__C = edict()
# Consumers can get config by: import config as cfg
cfg = __C

# Selected object
__C.TRACK_OBJ = 'Car'  # Pedestrian/Cyclist
if __C.TRACK_OBJ == 'Car' or __C.TRACK_OBJ == 'Van':
    __C.TEMPLATE_Z_MIN = -3.2
    __C.TEMPLATE_Z_MAX = 3.2
    __C.TEMPLATE_Y_MIN = -5.12
    __C.TEMPLATE_Y_MAX = 5.12
    __C.TEMPLATE_X_MIN = -3.2
    __C.TEMPLATE_X_MAX = 3.2
    __C.SCENE_Z_MIN = -5.12
    __C.SCENE_Z_MAX = 5.12
    __C.SCENE_Y_MIN = -5.12
    __C.SCENE_Y_MAX = 5.12
    __C.SCENE_X_MIN = -5.12
    __C.SCENE_X_MAX = 5.12
    __C.VOXEL_Z_SIZE = 0.16
    __C.VOXEL_Y_SIZE = 10.24
    __C.VOXEL_X_SIZE = 0.16
    __C.VOXEL_POINT_COUNT = 35
    __C.TEMPLATE_INPUT_DEPTH = int((__C.TEMPLATE_Z_MAX - __C.TEMPLATE_Z_MIN) / __C.VOXEL_Z_SIZE)
    __C.TEMPLATE_INPUT_HEIGHT = int((__C.TEMPLATE_Y_MAX - __C.TEMPLATE_Y_MIN) / __C.VOXEL_Y_SIZE)
    __C.TEMPLATE_INPUT_WIDTH = int((__C.TEMPLATE_X_MAX - __C.TEMPLATE_X_MIN) / __C.VOXEL_X_SIZE)
    __C.SCENE_INPUT_DEPTH = int((__C.SCENE_Z_MAX - __C.SCENE_Z_MIN) / __C.VOXEL_Z_SIZE)
    __C.SCENE_INPUT_HEIGHT = int((__C.SCENE_Y_MAX - __C.SCENE_Y_MIN) / __C.VOXEL_Y_SIZE)
    __C.SCENE_INPUT_WIDTH = int((__C.SCENE_X_MAX - __C.SCENE_X_MIN) / __C.VOXEL_X_SIZE)
    __C.INPUT_WIDTH = int(__C.SCENE_INPUT_WIDTH - __C.TEMPLATE_INPUT_WIDTH)
    __C.INPUT_HEIGHT = int(__C.SCENE_INPUT_HEIGHT - __C.TEMPLATE_INPUT_HEIGHT)
    __C.INPUT_DEPTH = int(__C.SCENE_INPUT_DEPTH - __C.TEMPLATE_INPUT_DEPTH)
    __C.FEATURE_RATIO = 2
    __C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO) + 1
    __C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO) + 1
    __C.FEATURE_DEPTH = int(__C.INPUT_DEPTH / __C.FEATURE_RATIO) + 1
else:
    __C.TEMPLATE_Z_MIN = -1.6
    __C.TEMPLATE_Z_MAX = 1.6
    __C.TEMPLATE_Y_MIN = -2.56
    __C.TEMPLATE_Y_MAX = 2.56
    __C.TEMPLATE_X_MIN = -1.6
    __C.TEMPLATE_X_MAX = 1.6
    __C.SCENE_Z_MIN = -2.56
    __C.SCENE_Z_MAX = 2.56
    __C.SCENE_Y_MIN = -2.56
    __C.SCENE_Y_MAX = 2.56
    __C.SCENE_X_MIN = -2.56
    __C.SCENE_X_MAX = 2.56
    __C.VOXEL_Z_SIZE = 0.08
    __C.VOXEL_Y_SIZE = 5.12
    __C.VOXEL_X_SIZE = 0.08
    __C.VOXEL_POINT_COUNT = 45
    __C.TEMPLATE_INPUT_DEPTH = int((__C.TEMPLATE_Z_MAX - __C.TEMPLATE_Z_MIN) / __C.VOXEL_Z_SIZE)
    __C.TEMPLATE_INPUT_HEIGHT = int((__C.TEMPLATE_Y_MAX - __C.TEMPLATE_Y_MIN) / __C.VOXEL_Y_SIZE)
    __C.TEMPLATE_INPUT_WIDTH = int((__C.TEMPLATE_X_MAX - __C.TEMPLATE_X_MIN) / __C.VOXEL_X_SIZE)
    __C.SCENE_INPUT_DEPTH = int((__C.SCENE_Z_MAX - __C.SCENE_Z_MIN) / __C.VOXEL_Z_SIZE)
    __C.SCENE_INPUT_HEIGHT = int((__C.SCENE_Y_MAX - __C.SCENE_Y_MIN) / __C.VOXEL_Y_SIZE)
    __C.SCENE_INPUT_WIDTH = int((__C.SCENE_X_MAX - __C.SCENE_X_MIN) / __C.VOXEL_X_SIZE)
    __C.INPUT_WIDTH = int(__C.SCENE_INPUT_WIDTH - __C.TEMPLATE_INPUT_WIDTH)
    __C.INPUT_HEIGHT = int(__C.SCENE_INPUT_HEIGHT - __C.TEMPLATE_INPUT_HEIGHT)
    __C.INPUT_DEPTH = int(__C.SCENE_INPUT_DEPTH - __C.TEMPLATE_INPUT_DEPTH)
    __C.FEATURE_RATIO = 2
    __C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO) + 1
    __C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO) + 1
    __C.FEATURE_DEPTH = int(__C.INPUT_DEPTH / __C.FEATURE_RATIO) + 1

if __C.TRACK_OBJ == 'Car' or __C.TRACK_OBJ == 'Van':
    # Car anchor
    __C.ANCHOR_L = 3.9
    __C.ANCHOR_W = 1.6
    __C.ANCHOR_H = 1.56
    __C.ANCHOR_Z = 0
    __C.RPN_POS_IOU = 0.6
    __C.RPN_NEG_IOU = 0.45
    __C.SEARCH_AREA = 2.0

elif __C.TRACK_OBJ == 'Pedestrian':
    # Pedestrian anchor
    __C.ANCHOR_L = 0.8
    __C.ANCHOR_W = 0.6
    __C.ANCHOR_H = 1.73
    __C.ANCHOR_Z = 0
    __C.RPN_POS_IOU = 0.7
    __C.RPN_NEG_IOU = 0.25
    __C.SEARCH_AREA = 0.3

elif __C.TRACK_OBJ == 'Cyclist':
    # Cyclist anchor
    __C.ANCHOR_L = 1.76
    __C.ANCHOR_W = 0.6
    __C.ANCHOR_H = 1.73
    __C.ANCHOR_Z = 0
    __C.RPN_POS_IOU = 0.65
    __C.RPN_NEG_IOU = 0.35
    __C.SEARCH_AREA = 0.6

if __name__ == '__main__':
    cfg.update({'TRACK_OBJ': 'aaa'})
    print('__C.TRACK = ' + __C.TRACK_OBJ)
