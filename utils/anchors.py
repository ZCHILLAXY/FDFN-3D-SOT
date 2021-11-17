import torch

import numpy as np
from utils.box_overlaps import *
from utils.anchor_utils import *

from config import cfg


def cal_anchors():
    # Output:
    # Anchors: (w, l, 2, 7) x y z h w l r
    x = np.linspace(cfg.SCENE_X_MIN, cfg.SCENE_X_MAX, cfg.FEATURE_WIDTH)
    y = np.linspace(cfg.SCENE_Y_MIN, cfg.SCENE_Y_MAX, cfg.FEATURE_WIDTH)
    z = np.linspace(cfg.SCENE_Z_MIN, cfg.SCENE_Z_MAX, cfg.FEATURE_WIDTH)
    cx, cy, cz = np.meshgrid(x, y, z)
    # All are (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.tile(cz[..., np.newaxis], 2)
    # cz = np.ones_like(cx) * cfg.ANCHOR_Z
    w = np.ones_like(cx) * cfg.ANCHOR_W
    l = np.ones_like(cy) * cfg.ANCHOR_L
    h = np.ones_like(cz) * cfg.ANCHOR_H
    # r = np.ones_like(cx)
    # r[..., 0] = 0  # 0
    # r[..., 1] = np.pi / 2

    # 7 * (w, l, 2) -> (w, l, 2, 7)
    anchors = np.stack([cx, cy, cz, w, l, h], axis=-1)

    return anchors


def cal_rpn_target(bbox, feature_map_shape, anchors, dim):
    # Input:
    #   labels: (N, N')
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)
    # Output:
    #   pos_equal_one (N, w, l, 2)
    #   neg_equal_one (N, w, l, 2)
    #   targets (N, w, l, 14)

    batch_size = bbox.shape[0]
    batch_gt_boxes3d = np.expand_dims(bbox, axis=1)

    # Defined in eq(1) in 2.2
    if dim == 'z':
        anchors_reshaped = anchors[:, :, 0, :, :].reshape(-1, 6)[:, [0, 1, 3, 4]]
    elif dim =='y':
        anchors_reshaped = anchors[:, :, 0, :, :].reshape(-1, 6)[:, [0, 1, 4, 5]]
    elif dim == 'x':
        anchors_reshaped = anchors[:, :, 0, :, :].reshape(-1, 6)[:, [0, 1, 3, 5]]
    anchors_d = np.sqrt(anchors_reshaped[:, 2] ** 2 + anchors_reshaped[:, 3] ** 2)

    pos_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    pos_equal_one[...] = -1
    smooth_labeling = 0.01
    targets = np.zeros((batch_size, *feature_map_shape, 8))
    depths = np.zeros((batch_size, *feature_map_shape, 4))

    for batch_id in range(batch_size):
        # BOTTLENECK; from (x,y,w,l) to (x1,y1,x2,y2)
        anchors_standup_2d = anchor_to_standup_box2d(anchors_reshaped)
        # BOTTLENECK
        gt_standup_2d = corner_to_standup_box2d(center_to_corner_box2d(
            batch_gt_boxes3d[batch_id], dim))
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        # iou = cal_box3d_iou(anchors_reshaped, batch_gt_boxes3d[batch_id])

        # Find anchor with highest iou (iou should also > 0)
        id_highest = np.argmax(iou.T, axis=1)
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # Find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > cfg.RPN_POS_IOU)
        # print(cfg.RPN_POS_IOU)

        # Find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < cfg.RPN_NEG_IOU, axis=1) == iou.shape[1])[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # Cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 1 * (1 - smooth_labeling) + smooth_labeling * (1 / 2) * \
                                                             iou[id_pos][0]

        # ATTENTION: index_z should be np.array
        if dim == 'z':
            targets[batch_id, index_x, index_y, np.array(index_z) * 4] = (
                                                                                 batch_gt_boxes3d[batch_id][
                                                                                     id_pos_gt, 0] -
                                                                                 anchors_reshaped[id_pos, 0]) / \
                                                                         anchors_d[
                                                                             id_pos]
            targets[batch_id, index_x, index_y, np.array(index_z) * 4 + 1] = (
                                                                                     batch_gt_boxes3d[batch_id][
                                                                                         id_pos_gt, 1] -
                                                                                     anchors_reshaped[
                                                                                         id_pos, 1]) / anchors_d[id_pos]
            targets[batch_id, index_x, index_y, np.array(index_z) * 4 + 2] = np.log(
                batch_gt_boxes3d[batch_id][id_pos_gt, 3] / anchors_reshaped[id_pos, 2])
            targets[batch_id, index_x, index_y, np.array(index_z) * 4 + 3] = np.log(
                batch_gt_boxes3d[batch_id][id_pos_gt, 4] / anchors_reshaped[id_pos, 3])
            depths[batch_id, index_x, index_y, np.array(index_z) * 2] = batch_gt_boxes3d[batch_id][id_pos_gt, 2]
            depths[batch_id, index_x, index_y, np.array(index_z) * 2 + 1] = batch_gt_boxes3d[batch_id][id_pos_gt, 5]
        elif dim == 'x':
            targets[batch_id, index_x, index_y, np.array(index_z) * 4] = (
                                                                                 batch_gt_boxes3d[batch_id][
                                                                                     id_pos_gt, 0] -
                                                                                 anchors_reshaped[id_pos, 0]) / \
                                                                         anchors_d[
                                                                             id_pos]
            targets[batch_id, index_x, index_y, np.array(index_z) * 4 + 1] = (
                                                                                     batch_gt_boxes3d[batch_id][
                                                                                         id_pos_gt, 2] -
                                                                                     anchors_reshaped[
                                                                                         id_pos, 1]) / anchors_d[id_pos]
            targets[batch_id, index_x, index_y, np.array(index_z) * 4 + 2] = np.log(
                batch_gt_boxes3d[batch_id][id_pos_gt, 3] / anchors_reshaped[id_pos, 2])
            targets[batch_id, index_x, index_y, np.array(index_z) * 4 + 3] = np.log(
                batch_gt_boxes3d[batch_id][id_pos_gt, 5] / anchors_reshaped[id_pos, 3])
            depths[batch_id, index_x, index_y, np.array(index_z) * 2] = batch_gt_boxes3d[batch_id][id_pos_gt, 1]
            depths[batch_id, index_x, index_y, np.array(index_z) * 2 + 1] = batch_gt_boxes3d[batch_id][id_pos_gt, 4]
        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 0 * (1 - smooth_labeling) + smooth_labeling * (1 / 2) * \
                                                             iou[id_neg][0]
        # To avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(id_highest, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 1 * (1 - smooth_labeling) + smooth_labeling * (1 / 2) * \
                                                             iou[id_highest][0]
    return pos_equal_one, targets, depths

