import torch
import os
import copy
import numpy as np
from pyquaternion import Quaternion
from utils.data_classes import PointCloud
from utils.metrics import estimateOverlap
from config import cfg
from scipy.optimize import leastsq


def distanceBB_Gaussian(box1, box2, sigma=1):
    off1 = np.array([
        box1.center[0], box1.center[2],
        Quaternion(matrix=box1.rotation_matrix).degrees
    ])
    off2 = np.array([
        box2.center[0], box2.center[2],
        Quaternion(matrix=box2.rotation_matrix).degrees
    ])
    dist = np.linalg.norm(off1 - off2)
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return score


# IoU or Gaussian score map
def getScoreGaussian(offset, sigma=1):
    coeffs = [1, 1, 1 / 5]
    dist = np.linalg.norm(np.multiply(offset, coeffs))
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return torch.tensor([score])


def getScoreIoU(a, b):
    score = estimateOverlap(a, b)
    return torch.tensor([score])


def getScoreHingeIoU(a, b):
    score = estimateOverlap(a, b)
    if score < 0.5:
        score = 0.0
    return torch.tensor([score])


def getOffsetBB(box, offset):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    # REMOVE TRANSfORM
    if len(offset) == 3:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
    elif len(offset) == 4:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[3] * np.pi / 180))
    if offset[0]>new_box.wlh[0]:
        offset[0] = np.random.uniform(-1,1)
    if offset[1]>min(new_box.wlh[1],2):
        offset[1] = np.random.uniform(-1,1)
    new_box.translate(np.array([offset[0], offset[1], 0]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def voxelize(PC, dim_size=[48, 108, 48]):
    # PC = normalizePC(PC)
    if np.isscalar(dim_size):
        dim_size = [dim_size] * 3
    dim_size = np.atleast_2d(dim_size).T
    PC = (PC + 0.5) * dim_size
    # truncate to integers
    xyz = PC.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dim_size), 0)
    xyz = xyz[:, valid_ix]
    out = np.zeros(dim_size.flatten(), dtype=np.float32)
    out[tuple(xyz)] = 1
    # print(out)
    return out


# def regularizePC2(input_size, PC,):
#     return regularizePC(PC=PC, input_size=input_size)


def regularizePC_template(PC,input_size,istrain=True):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 0:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != int(input_size/2):
            if not istrain:
                np.random.seed(1)
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=int(input_size/2), dtype=np.int64)
            PC = PC[:, new_pts_idx]
        PC = PC.reshape((3, int(input_size/2))).T

    else:
        PC = np.zeros((3, int(input_size/2))).T

    return PC

def regularizePC_scene(PC,input_size,istrain=True):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 0:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != input_size:
            if not istrain:
                np.random.seed(1)
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
            
        PC = PC.reshape((3, input_size)).T
        
    else:
        PC = np.zeros((3, input_size)).T

    return PC

def getModel(PCs, boxes, offset=0, scale=1.0, normalize=False):

    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = np.ones((PCs[0].points.shape[0], 0))

    for PC, box in zip(PCs, boxes):
        cropped_PC = cropAndCenterPC(
            PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.points.shape[1] > 0:
            points = np.concatenate([points, cropped_PC.points], axis=1)

    PC = PointCloud(points)

    return PC


def cropPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC

def getlabelPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate((rot_mat))
    box_tmp.rotate(Quaternion(matrix=(rot_mat)))
    
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_label = np.zeros(new_PC.points.shape[1])
    new_label[close] = 1
    return new_label

def cropPCwithlabel(PC, box, label, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    new_label = label[close]
    return new_PC,new_label

def weight_process(include,low,high):
    if include<low:
        weight = 0.7
    elif include >high:
        weight = 1
    else:
        weight = (include*2.0+3.0*high-5.0*low)/(5*(high-low))
    return weight

def func(a, x):
    k, b = a
    return k * x + b
def dist(a, x, y):
    return func(a, x) - y

def weight_process2(k):
    k = abs(k)
    if k>1:
        weight = 0.7
    else:
        weight = 1-0.3*k
    return weight



def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset, scale=scale)


    if normalize:
        new_PC.normalize(box.wlh)

    return new_PC

def Centerbox(sample_box, gt_box):
    rot_mat = np.transpose(gt_box.rotation_matrix)
    trans = -gt_box.center

    new_box = copy.deepcopy(sample_box)
    new_box.translate(trans)
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    return new_box


def cropAndCenterPC_new(PC, sample_box, gt_box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    new_label = getlabelPC(new_PC, gt_box, offset=offset, scale=scale)
    new_box_gt = copy.deepcopy(gt_box)
    # new_box_gt2 = copy.deepcopy(gt_box)

    #rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_box_gt.translate(trans)
    new_box_gt.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+cfg.SEARCH_AREA, scale=1 * scale)
    #new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+0.6, scale=1 * scale)

    center_box_gt = [new_box_gt.center[0],new_box_gt.center[1],new_box_gt.center[2],new_box_gt.wlh[0],new_box_gt.wlh[1],new_box_gt.wlh[2],new_box_gt.orientation.axis[2] * new_box_gt.orientation.radians]
    center_box_gt = np.array(center_box_gt)
    # label_reg = np.tile(label_reg,[np.size(new_label),1])

    if normalize:
        new_PC.normalize(sample_box.wlh)
    return new_PC, center_box_gt, trans, rot_mat

# def cropAndCenterPC_label_test_time(PC, sample_box, offset=0, scale=1.0):
#
#     new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)
#
#     new_box = copy.deepcopy(sample_box)
#
#     rot_quat = Quaternion(matrix=new_box.rotation_matrix)
#     rot_mat = np.transpose(new_box.rotation_matrix)
#     trans = -new_box.center
#
#     # align data
#     new_PC.translate(trans)
#     new_box.translate(trans)
#     new_PC.rotate((rot_mat))
#     new_box.rotate(Quaternion(matrix=(rot_mat)))
#
#     # crop around box
#     new_PC = cropPC(new_PC, new_box, offset=offset+2.0, scale=scale)
#
#     return new_PC

# def cropAndCenterPC_test(PC, sample_box, gt_box, offset=0, scale=1.0, normalize=False):
#
#     new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)
#
#     new_box = copy.deepcopy(sample_box)
#
#     new_label = getlabelPC(new_PC, gt_box, offset=offset, scale=scale)
#     new_box_gt = copy.deepcopy(gt_box)
#     # new_box_gt2 = copy.deepcopy(gt_box)
#
#     rot_quat = Quaternion(matrix=new_box.rotation_matrix)
#     rot_mat = np.transpose(new_box.rotation_matrix)
#     trans = -new_box.center
#
#     # align data
#     new_PC.translate(trans)
#     new_box.translate(trans)
#     new_PC.rotate((rot_mat))
#     new_box.rotate(Quaternion(matrix=(rot_mat)))
#
#     new_box_gt.translate(trans)
#     new_box_gt.rotate(Quaternion(matrix=(rot_mat)))
#     # new_box_gt2.translate(trans)
#     # new_box_gt2.rotate(rot_quat.inverse)
#
#     # crop around box
#     new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+cfg.SEARCH_AREA, scale=1 * scale)
#     #new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+0.6, scale=1 * scale)
#
#     label_reg = [new_box_gt.center[0],new_box_gt.center[1],new_box_gt.center[2]]
#     label_reg = np.array(label_reg)
#     # label_reg = (new_PC.points - np.tile(new_box_gt.center,[np.size(new_label),1]).T) * np.expand_dims(new_label, axis=0)
#     # newoff = [new_box_gt.center[0],new_box_gt.center[1],new_box_gt.center[2]]
#     # newoff = np.array(newoff)
#
#     if normalize:
#         new_PC.normalize(sample_box.wlh)
#     return new_PC, trans, rot_mat



def getPJMatrix(calib):
    R = np.zeros([4, 4], dtype=np.float32)
    R[:3, :3] = calib['R_rect']
    R[3, 3] = 1
    M = np.dot(calib['P2:'], R)
    return M


def project_velo2rgb(box, pj_matrix):

    box = box.corners().T
    box3d = np.ones([8, 4], dtype=np.float32)
    box3d[:, :3] = box
    box2d = np.dot(pj_matrix, box3d.T)
    box2d = box2d[:2, :].T/box2d[2, :].reshape(8, 1)
    projections = box2d

    minx = 0 if np.min(projections[:, 0]) < 0 else int(np.min(projections[:, 0]))
    maxx = 0 if np.max(projections[:, 0]) < 0 else int(np.max(projections[:, 0]))
    miny = 0 if np.min(projections[:, 1]) < 0 else int(np.min(projections[:, 1]))
    maxy = 0 if np.max(projections[:, 1]) < 0 else int(np.max(projections[:, 1]))

    rgb_box = [minx, miny, maxx, maxy]

    return projections, rgb_box


def cropRGB_train(RGB, box, pj_matrix, search=False, offset=0, scale=1.0):

    box_tmp = copy.deepcopy(box)
    if search:
        crop_offset = offset + cfg.SEARCH_AREA
    else:
        crop_offset = offset

    box_tmp.wlh = box_tmp.wlh + [crop_offset, crop_offset, crop_offset]
    box_tmp.wlh = box_tmp.wlh * scale


    # maxi = np.max(box_tmp.corners(), 1) + offset
    # mini = np.min(box_tmp.corners(), 1) - offset

    box_projection, box_rgb = project_velo2rgb(box_tmp, pj_matrix)
    new_RGB = RGB[box_rgb[1]: box_rgb[3], box_rgb[0]: box_rgb[2]]

    return new_RGB

def cropForwardPC(PC, ORI_RGB, CP_RGB, box, pj_matrix, search=False, offset=0, scale=1.0):
    points = PC.points.T[:, 0:3]  # lidar xyz (front, left, up)
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[1, :] < 0), axis=1)
    cam = np.dot(pj_matrix, velo)
    cam[:2] /= cam[2, :]

    IMG_H, IMG_W, _ = ORI_RGB.shape

    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)


    box_tmp = copy.deepcopy(box)
    if search:
        crop_offset = offset + cfg.SEARCH_AREA
    else:
        crop_offset = offset

    box_tmp.wlh = box_tmp.wlh + [crop_offset, crop_offset, crop_offset]
    box_tmp.wlh = box_tmp.wlh * scale
    box_projection, box_rgb = project_velo2rgb(box_tmp, pj_matrix)

    center = [-box_rgb[0], -box_rgb[1]]

    u, v, z = cam
    u_out = np.logical_or(u < box_rgb[0], u > box_rgb[2])
    v_out = np.logical_or(v < box_rgb[1], v > box_rgb[3])
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)

    for i in range(2):
        cam[i, :] = cam[i, :] + center[i]

    if search:
        height_ratio = CP_RGB.shape[0] / cfg.SCENE_INPUT_WIDTH
        width_ratio = CP_RGB.shape[1] / cfg.SCENE_INPUT_WIDTH
    else:
        height_ratio = CP_RGB.shape[0] / cfg.SCENE_INPUT_WIDTH
        width_ratio = CP_RGB.shape[1] / cfg.TEMPLATE_INPUT_WIDTH

    cam[0, :] = cam[0, :] / width_ratio
    cam[1, :] = cam[1, :] / height_ratio

    u, v, z = cam
    if search:
        u_out = np.logical_or(u < 0, u > cfg.SCENE_INPUT_WIDTH - 1)
        v_out = np.logical_or(v < 0, v > cfg.SCENE_INPUT_WIDTH - 1)
    else:
        u_out = np.logical_or(u < 0, u > cfg.SCENE_INPUT_WIDTH - 1)
        v_out = np.logical_or(v < 0, v > cfg.TEMPLATE_INPUT_WIDTH - 1)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)

    cam = PointCloud(cam)

    return cam

def cropRGB_test(RGB, box, pj_matrix, search=False, offset=0, scale=1.0):

    box_tmp = copy.deepcopy(box)
    if search:
        crop_offset = offset + cfg.SEARCH_AREA
    else:
        crop_offset = offset

    box_tmp.wlh = box_tmp.wlh + [crop_offset, crop_offset, crop_offset]
    box_tmp.wlh = box_tmp.wlh * scale


    # maxi = np.max(box_tmp.corners(), 1) + offset
    # mini = np.min(box_tmp.corners(), 1) - offset

    box_projection, box_rgb = project_velo2rgb(box_tmp, pj_matrix)
    new_RGB = RGB[box_rgb[1]: box_rgb[3], box_rgb[0]: box_rgb[2]]

    if new_RGB.shape[0] == 0 or new_RGB.shape[1] == 0:
        new_RGB = np.zeros((1, 1, 3))

    return new_RGB
