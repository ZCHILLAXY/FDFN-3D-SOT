import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import CAM_Module, CrossCAM_Module, PAM_Module, FPAM_Module, FCAM_Module, YFCAM_Module, \
    Dy_FCAM_Module, Dy_FPAM_Module
from model.c_a_output import Center_Angle_Output_Module
from model.pointpillar.base_bev_backbone import BaseBEVBackbone
from model.pointpillar.pillar_vfe import PillarVFE
from model.pointpillar.pointpillar_scatter import PointPillarScatter
from model.pr_res import PRFuseBlock
from model.pr_trans import PSViT
from model.rgb_pillar.rgb_scatter import RGBScatter
from model.rgb_pillar.rgb_vfe import RGBVFE
from utils.anchors import *
from utils.colorize import colorize
from model.rpn import MiddleAndRPNFeature, ConvMD
from model.rpn_deep import RPN_Deep
from config import cfg
import pdb

from model.prroi_pool import PrRoIPool2D
from utils.common_utils import project_velo2rgb

from utils.data_classes import PointCloud, Box
from pyquaternion import Quaternion

small_addon_for_BCE = 1e-6


class SiamPillar(nn.Module):
    def __init__(self):
        super(SiamPillar, self).__init__()

        self.template_pvfe = PillarVFE(template=True)
        self.scene_pvfe = PillarVFE()
        self.template_scatter = PointPillarScatter(template=True)
        self.scene_scatter = PointPillarScatter()
        self.rgb_template_pvfe = RGBVFE(template=True)
        self.rgb_template_scatter = RGBScatter(template=True)
        self.rgb_scene_pvfe = RGBVFE()
        self.rgb_scene_scatter = RGBScatter()

        self.template_rgb_extract = PRFuseBlock()
        self.scene_rgb_extract = PRFuseBlock()

        self.template_rpn = BaseBEVBackbone()
        self.scene_rpn = BaseBEVBackbone()

        # self.dy_fcam = Dy_FCAM_Module()
        self.dy_fpam = Dy_FPAM_Module()

        self.psvit = PSViT()

        # self.RPN_1 = RPN_Deep()
        # self.RPN_2 = RPN_Deep()
        # self.RPN_3 = RPN_Deep()

        self.module_cls = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 2 * 2, kernel_size=1, padding=0, stride=1), ConvMD(2, 2*2, 2*2, 1, (1, 1), (0, 0), bn=False, activation=False))
        self.module_reg = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 4 * 2, kernel_size=1, padding=0, stride=1), ConvMD(2, 4*2, 4*2, 1, (1, 1), (0, 0), bn=False, activation=False))
        self.module_depth = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 2 * 2, kernel_size=1, padding=0, stride=1), ConvMD(2, 2*2, 2*2, 1, (1, 1), (0, 0), bn=False, activation=False))


        # self.weighted_sum_layer_alpha = nn.Conv2d(3 * 4, 4, kernel_size=1, padding=0,
        #     groups=4)
        # self.weighted_sum_layer_beta = nn.Conv2d(3 * 8, 8, kernel_size=1, padding=0,
        #     groups=8)
        # self.weighted_sum_layer_gama = nn.Conv2d(3 * 4, 4, kernel_size=1, padding=0,
        #     groups=4)


        # self.prob_conv = ConvMD(2, 2*2, 2*2, 1, (1, 1), (0, 0), bn=False, activation=False)
        # self.reg_conv = ConvMD(2, 4*2, 4*2, 1, (1, 1), (0, 0), bn=False, activation=False)
        # self.depth_conv = ConvMD(2, 2*2, 2*2, 1, (1, 1), (0, 0), bn=False, activation=False)


        # Generate anchors
        self.anchors = cal_anchors()    # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 8]; 2 means two rotations; 8 means (cx, cy, cz, h, w, l, r)

        self.pr_pooling = PrRoIPool2D(40, 40, 1.0)

        self.crosscam = CrossCAM_Module()
        self.output_layer = Center_Angle_Output_Module()


    def forward(self, batch_size, t_vox_feature, t_vox_number, t_vox_coordinate,
                s_vox_feature, s_vox_number, s_vox_coordinate,
                rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate,
                rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate,
                gt_RGB, sample_RGB, template_box, sample_box):
        '''

        :param batch_size:
        :param t_vox_feature:
        :param t_vox_number:
        :param t_vox_coordinate:
        :param s_vox_feature:
        :param s_vox_number:
        :param s_vox_coordinate:
        :param rgb_t_vox_feature:
        :param rgb_t_vox_number:
        :param rgb_t_vox_coordinate:
        :param rgb_s_vox_feature:
        :param rgb_s_vox_number:
        :param rgb_s_vox_coordinate:
        :param gt_RGB:
        :param sample_RGB:
        :param template_box:
        :param sample_box:
        :return:
        '''
        template_pvfes = self.template_pvfe(t_vox_feature, t_vox_number, t_vox_coordinate, template_box)
        template_features = self.template_scatter(template_pvfes, t_vox_coordinate, batch_size)
        scene_pvfes = self.scene_pvfe(s_vox_feature, s_vox_number, s_vox_coordinate, sample_box)
        scene_features = self.scene_scatter(scene_pvfes, s_vox_coordinate, batch_size)
        rgb_template_pvfes = self.rgb_template_pvfe(rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate)
        rgb_template_features = self.rgb_template_scatter(rgb_template_pvfes, rgb_t_vox_coordinate, batch_size)
        rgb_scene_pvfes = self.rgb_scene_pvfe(rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate)
        rgb_scene_features = self.rgb_scene_scatter(rgb_scene_pvfes, rgb_s_vox_coordinate, batch_size)

        rgb_template_features = self.template_rgb_extract(rgb_template_features, gt_RGB.permute(0, 3, 1, 2))
        rgb_scene_features = self.scene_rgb_extract(rgb_scene_features, sample_RGB.permute(0, 3, 1, 2))

        template_features_att, scene_features_att = self.dy_fpam(rgb_template_features, rgb_scene_features, template_features,
                                                          scene_features)
        correlation_sample = self.psvit(scene_features_att)

        # template_out_1, template_out_2, template_out_3 = self.template_rpn(template_features_att)
        # scene_out_1, scene_out_2, scene_out_3 = self.scene_rpn(scene_features_att)
        #
        # cls_prediction_1, regression_prediction_1, depth_prediction_1 = self.RPN_1(
        #     template_out_1, scene_out_1)
        # cls_prediction_2, regression_prediction_2, depth_prediction_2 = self.RPN_2(
        #     template_out_2, scene_out_2)
        # cls_prediction_3, regression_prediction_3, depth_prediction_3 = self.RPN_3(
        #     template_out_3, scene_out_3)
        #
        #
        # stacked_cls_prediction = torch.cat((cls_prediction_1, cls_prediction_2, cls_prediction_3),
        # 2).reshape(batch_size, 4, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH).reshape(batch_size, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH)
        # stacked_regression_prediction = torch.cat((regression_prediction_1, regression_prediction_2, regression_prediction_3),
        # 2).reshape(batch_size, 8, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH).reshape(batch_size, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH)
        # stacked_depth_prediction = torch.cat((depth_prediction_1, depth_prediction_2, depth_prediction_3),
        # 2).reshape(batch_size, 4, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH).reshape(batch_size, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH)

        p_map = self.module_cls(correlation_sample)
        r_map = self.module_reg(correlation_sample)
        d_map = self.module_depth(correlation_sample)

        pred_conf = p_map.reshape(-1, 2, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_WIDTH).permute(0, 2, 1)
        pred_reg = r_map.reshape(-1, 4, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_WIDTH).permute(0, 2, 1)
        pred_depth = d_map.reshape(-1, 2, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_WIDTH).permute(0, 2, 1)

        batch_boxes2d = delta_to_boxes2d(pred_reg, self.anchors, dim='x')
        probs = F.sigmoid(pred_conf)[:, :, 1].squeeze()
        rois = []
        rois_center = []

        for batch_id in range(batch_size):
            best = torch.argmax(probs[batch_id, :])
            regression = batch_boxes2d[batch_id, best, :].detach().cpu().numpy()
            center_y = pred_depth[batch_id, best, :].detach().cpu().numpy()[0]
            depth = pred_depth[batch_id, best, :].detach().cpu().numpy()[1]
            x = (regression[0] - regression[2] / 2 - cfg.SCENE_X_MIN) / cfg.VOXEL_X_SIZE
            z = (regression[1] - regression[3] / 2 - cfg.SCENE_Z_MIN) / cfg.VOXEL_Z_SIZE
            w = regression[2] / cfg.VOXEL_X_SIZE
            h = regression[3] / cfg.VOXEL_Z_SIZE
            center_x = regression[0]
            center_z = regression[1]

            roi = torch.from_numpy(np.array([batch_id, x, z, x + w, z + h])).cuda().float()
            roi_center = torch.from_numpy(np.array([batch_id, center_x, center_y, center_z])).cuda().float()
            rois.append(roi)
            rois_center.append(roi_center)
        rois = torch.cat(rois).reshape(batch_size, -1)
        rois_center = torch.cat(rois_center).reshape(batch_size, -1)
        roi_features = self.pr_pooling(scene_features_att, rois)
        object_semantic = self.crosscam(template_features_att, roi_features)
        final_coord, final_angle = self.output_layer(object_semantic, rois_center[:, 1:])

        return pred_conf, pred_reg, pred_depth, final_coord, final_angle


    def track_init(self, t_vox_feature, t_vox_number, t_vox_coordinate,
                        rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate,
                        gt_RGB, template_box, box_wlh):

        template_pvfes = self.template_pvfe(t_vox_feature, t_vox_number, t_vox_coordinate, template_box)
        rgb_template_pvfes = self.rgb_template_pvfe(rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate)

        self.template_features = self.template_scatter(template_pvfes, t_vox_coordinate, 1)
        rgb_template_features = self.rgb_template_scatter(rgb_template_pvfes, rgb_t_vox_coordinate, 1)
        self.rgb_template_features = self.template_rgb_extract(rgb_template_features, gt_RGB.permute(0, 3, 1, 2))
        self.box_wlh = box_wlh


    def track(self, s_vox_feature, s_vox_number, s_vox_coordinate,
                    rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate,
                    candidate_RGB, sample_box):
        
        scene_pvfes = self.scene_pvfe(s_vox_feature, s_vox_number, s_vox_coordinate, sample_box)
        scene_features = self.scene_scatter(scene_pvfes, s_vox_coordinate, 1)
        rgb_scene_pvfes = self.rgb_scene_pvfe(rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate)
        rgb_scene_features = self.rgb_scene_scatter(rgb_scene_pvfes, rgb_s_vox_coordinate, 1)
        rgb_scene_features = self.scene_rgb_extract(rgb_scene_features, candidate_RGB.permute(0, 3, 1, 2))

        template_features_att, scene_features_att = self.dy_fpam(self.rgb_template_features, rgb_scene_features, self.template_features,
                                                          scene_features)

        correlation_sample = self.psvit(scene_features_att)

        # template_out_1, template_out_2, template_out_3 = self.template_rpn(template_features_att)
        # scene_out_1, scene_out_2, scene_out_3 = self.scene_rpn(scene_features_att)
        #
        # cls_prediction_1, regression_prediction_1, depth_prediction_1 = self.RPN_1(
        #     template_out_1, scene_out_1)
        # cls_prediction_2, regression_prediction_2, depth_prediction_2 = self.RPN_2(
        #     template_out_2, scene_out_2)
        # cls_prediction_3, regression_prediction_3, depth_prediction_3 = self.RPN_3(
        #     template_out_3, scene_out_3)
        #
        # stacked_cls_prediction = torch.cat((cls_prediction_1, cls_prediction_2, cls_prediction_3),
        # 2).reshape(1, 4, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH).reshape(1, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH)
        # stacked_regression_prediction = torch.cat((regression_prediction_1, regression_prediction_2, regression_prediction_3),
        # 2).reshape(1, 8, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH).reshape(1, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH)
        # stacked_depth_prediction = torch.cat((depth_prediction_1, depth_prediction_2, depth_prediction_3),
        # 2).reshape(1, 4, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH).reshape(1, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH)
        # pillar_cls_prediction = self.weighted_sum_layer_alpha(stacked_cls_prediction)
        # pillar_reg_prediction = self.weighted_sum_layer_beta(stacked_regression_prediction)
        # pillar_depth_prediction = self.weighted_sum_layer_gama(stacked_depth_prediction)
        #
        p_map = self.module_cls(correlation_sample)
        r_map = self.module_reg(correlation_sample)
        d_map = self.module_depth(correlation_sample)

        pred_conf = p_map.reshape(-1, 2, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_WIDTH).permute(0, 2, 1)
        pred_reg = r_map.reshape(-1, 4, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_WIDTH).permute(0, 2, 1)
        pred_depth = d_map.reshape(-1, 2, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_WIDTH).permute(0, 2, 1)

        batch_boxes2d = delta_to_boxes2d(pred_reg, self.anchors, dim='x')
        probs = F.sigmoid(pred_conf)[:, :, 1]

        best = torch.argmax(probs).squeeze()

        regression = batch_boxes2d[0, best, :].detach().cpu().numpy()
        center_y = pred_depth[0, best, :].detach().cpu().numpy()[0]
        depth = pred_depth[0, best, :].detach().cpu().numpy()[1]
        x = (regression[0] - regression[2] / 2 - cfg.SCENE_X_MIN) / cfg.VOXEL_X_SIZE
        z = (regression[1] - regression[3] / 2 - cfg.SCENE_Z_MIN) / cfg.VOXEL_Z_SIZE
        w = regression[2] / cfg.VOXEL_X_SIZE
        h = regression[3] / cfg.VOXEL_Z_SIZE
        center_x = regression[0]
        center_z = regression[1]
        roi = torch.from_numpy(np.array([0, x, z, x + w, z + h])).cuda().float().reshape(1, -1)
        roi_center = torch.from_numpy(np.array([0, center_x, center_y, center_z])).cuda().float().reshape(1, -1)
        roi_features = self.pr_pooling(scene_features_att, roi)

        object_semantic = self.crosscam(template_features_att, roi_features)
        final_coord, final_angle = self.output_layer(object_semantic, roi_center[:, 1:])

        final_coord = final_coord.squeeze().detach().cpu().numpy()
        final_angle = final_angle.squeeze().detach().cpu().numpy()

        ret_box3d = [final_coord[0], final_coord[1], final_coord[2], self.box_wlh[0], self.box_wlh[1], self.box_wlh[2], final_angle]

        return ret_box3d



