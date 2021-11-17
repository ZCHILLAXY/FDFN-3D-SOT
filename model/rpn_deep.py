#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

...
'''=================================================
@Project -> File   ：SiamVoxel -> rpn_deep.py
@Author ：Yi_Zhuang
@Time   ：2020/5/16 3:40 下午
=================================================='''


class RPN_Deep(nn.Module):
    def __init__(self):
        super(RPN_Deep, self).__init__()
        self.exam_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))
        self.inst_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))
        self.exam_reg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))
        self.inst_reg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))
        self.exam_depth = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))
        self.inst_depth = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))

        # self.exam_cls = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256))
        # self.inst_cls = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256))
        # self.exam_reg = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256))
        # self.inst_reg = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256))




        self.fusion_module_cls = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_reg = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_depth = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))


        self.Box_Head = nn.Sequential(
            nn.Conv2d(256, 4*2, kernel_size=1, padding=0, stride=1))
        self.Cls_Head = nn.Sequential(
            nn.Conv2d(256, 2*2, kernel_size=1, padding=0, stride=1))
        self.Depth_Head = nn.Sequential(
             nn.Conv2d(256, 2*2, kernel_size=1, padding=0, stride=1))


    def forward(self, examplar_feature_map, instance_feature_map):
        exam_cls_output = self.exam_cls(examplar_feature_map)
        exam_cls_output = exam_cls_output.reshape(-1, 18, 18)
        exam_cls_output = exam_cls_output.unsqueeze(0).permute(1, 0, 2, 3)

        inst_cls_output = self.inst_cls(instance_feature_map)
        inst_cls_output = inst_cls_output.reshape(-1, 30, 30)
        inst_cls_output = inst_cls_output.unsqueeze(0)

        exam_reg_output = self.exam_reg(examplar_feature_map)
        exam_reg_output = exam_reg_output.reshape(-1, 18, 18)
        exam_reg_output = exam_reg_output.unsqueeze(0).permute(1, 0, 2, 3)

        inst_reg_output = self.inst_reg(instance_feature_map)
        inst_reg_output = inst_reg_output.reshape(-1, 30, 30)
        inst_reg_output = inst_reg_output.unsqueeze(0)

        exam_depth_output = self.exam_depth(examplar_feature_map)
        exam_depth_output = exam_depth_output.reshape(-1, 18, 18)
        exam_depth_output = exam_depth_output.unsqueeze(0).permute(1, 0, 2, 3)

        inst_depth_output = self.inst_depth(instance_feature_map)
        inst_depth_output = inst_depth_output.reshape(-1, 30, 30)
        inst_depth_output = inst_depth_output.unsqueeze(0)



        depthwise_cross_cls = F.conv2d(
            inst_cls_output, exam_cls_output, bias=None, stride=1, padding=0, groups=exam_cls_output.size()[0]).squeeze()
        depthwise_cross_reg = F.conv2d(
            inst_reg_output, exam_reg_output, bias=None, stride=1, padding=0, groups=exam_reg_output.size()[0]).squeeze()
        depthwise_cross_depth = F.conv2d(
            inst_depth_output, exam_depth_output, bias=None, stride=1, padding=0, groups=exam_reg_output.size()[0]).squeeze()
        depthwise_cross_cls = depthwise_cross_cls.reshape(-1, 256, 13, 13)
        depthwise_cross_reg = depthwise_cross_reg.reshape(-1, 256, 13, 13)
        depthwise_cross_depth = depthwise_cross_depth.reshape(-1, 256, 13, 13)

        # depthwise_cross_seed = F.conv2d(
        #     inst_seed_output, exam_seed_output, bias=None, stride=1, padding=0, groups=exam_seed_output.size()[0]).squeeze()
        # depthwise_cross_offset = F.conv2d(
        #     inst_offset_output, exam_offset_output, bias=None, stride=1, padding=0, groups=exam_offset_output.size()[0]).squeeze()

        # depthwise_cross = F.conv2d(
        #      inst_output, exam_output, bias=None, stride=1, padding=0, groups=exam_output.size()[0]).squeeze()



        # depthwise_cross_cls = depthwise_cross_cls.reshape(-1, 256, 13, 13)
        # depthwise_cross_reg = depthwise_cross_reg.reshape(-1, 256, 13, 13)
        # depthwise_cross_seed = depthwise_cross_seed.reshape(-1, 256, 13, 13)
        # depthwise_cross_offset = depthwise_cross_offset.reshape(-1, 256, 13, 13)


        depthwise_cross_cls = self.fusion_module_cls(depthwise_cross_cls)
        depthwise_cross_reg = self.fusion_module_reg(depthwise_cross_reg)
        depthwise_cross_depth = self.fusion_module_depth(depthwise_cross_depth)

        cls_prediction = self.Cls_Head(depthwise_cross_cls)
        box_regression_prediction = self.Box_Head(depthwise_cross_reg)
        depth_prediction = self.Depth_Head(depthwise_cross_depth)

        return cls_prediction, box_regression_prediction, depth_prediction
