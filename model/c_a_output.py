import torch

import torch.nn as nn
import torch.nn.functional as F

from dcn.dcn_v2 import DCN


class Center_Angle_Output_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(Center_Angle_Output_Module, self).__init__()

        self.x_linear = nn.Sequential(nn.Linear(in_features=40, out_features=1))
        self.y_linear = nn.Sequential(nn.Linear(in_features=40, out_features=1))
        self.z_linear = nn.Sequential(nn.Linear(in_features=64, out_features=1))

        self.angle_output = nn.Sequential(DCN(64, 1, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(40, stride=1))

    def forward(self, semantic_map, roi_center):
        m_batchsize, C, h, w = semantic_map.size()

        x_zplane = F.avg_pool2d(semantic_map.permute(0, 2, 3, 1), kernel_size=[w, C]).reshape(m_batchsize, -1)
        y_zplane = F.avg_pool2d(semantic_map.permute(0, 3, 2, 1), kernel_size=[h, C]).reshape(m_batchsize, -1)
        z_zplane = F.avg_pool2d(semantic_map, kernel_size=[w, h]).reshape(m_batchsize, -1)

        x_bias = self.x_linear(x_zplane)
        y_bias = self.y_linear(y_zplane)
        z_bias = self.z_linear(z_zplane)

        final_center = torch.cat([x_bias, z_bias, y_bias], dim=1) + roi_center
        final_angle = self.angle_output(semantic_map).reshape(m_batchsize, -1)


        return final_center, final_angle