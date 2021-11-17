#!/usr/bin/python3
"""
@Project: SiamPillar 
@File: attention.py
@Author: Zhuang Yi
@Date: 2020/9/12
"""
import torch

import torch.nn as nn
import torch.nn.functional as F


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim=256):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, template_map, scene_map):
        m_batchsize, C, t_h, t_w = template_map.size()
        _, _, s_h, s_w = scene_map.size()

        t_proj_query_c = template_map.view(m_batchsize, C, -1)
        t_proj_key_c = template_map.view(m_batchsize, C, -1).permute(0, 2, 1)
        t_energy_c = torch.bmm(t_proj_query_c, t_proj_key_c)
        t_energy_c_new = torch.max(t_energy_c, -1, keepdim=True)[0].expand_as(t_energy_c) - t_energy_c
        t_attention_c = self.softmax(t_energy_c_new)
        t_proj_value_c = template_map.view(m_batchsize, C, -1)

        t_out_c = torch.bmm(t_attention_c, t_proj_value_c)
        t_out_c = t_out_c.view(m_batchsize, C, t_h, t_w)
        t_out_c = self.alpha * t_out_c + template_map


        s_proj_query_c = scene_map.view(m_batchsize, C, -1)
        s_proj_key_c = scene_map.view(m_batchsize, C, -1).permute(0, 2, 1)
        s_energy_c = torch.bmm(s_proj_query_c, s_proj_key_c)
        s_energy_c_new = torch.max(s_energy_c, -1, keepdim=True)[0].expand_as(s_energy_c) - s_energy_c
        s_attention_c = self.softmax(s_energy_c_new)
        s_proj_value_c = scene_map.view(m_batchsize, C, -1)

        s_out_c = torch.bmm(s_attention_c, s_proj_value_c)
        s_out_c = s_out_c.view(m_batchsize, C, s_h, s_w)
        s_out_c = self.beta * s_out_c + scene_map

        t_out_x = torch.bmm(s_attention_c, t_proj_value_c)
        t_out_x = t_out_x.view(m_batchsize, C, t_h, t_w)
        t_out_x = self.gamma * t_out_x + template_map

        s_out_x = torch.bmm(t_attention_c, s_proj_value_c)
        s_out_x = s_out_x.view(m_batchsize, C, s_h, s_w)
        s_out_x = self.omega * s_out_x + scene_map

        fused_template_map = t_out_c + t_out_x
        fused_scene_map = s_out_c + s_out_x


        return fused_template_map, fused_scene_map


class PAM_Module(nn.Module):

    def __init__(self, in_dim=64):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.t_query_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.t_query_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, template_map, scene_map):

        m_batchsize, C, t_h, t_w = template_map.size()
        _, _, s_h, s_w = scene_map.size()

        t_proj_query_w = self.t_query_conv_w(template_map).view(m_batchsize, -1, t_w).permute(0, 2, 1)
        t_proj_key_w = self.t_key_conv_w(template_map).view(m_batchsize, -1, t_w)
        t_energy_w = torch.bmm(t_proj_query_w, t_proj_key_w)
        t_attention_w = self.softmax(t_energy_w)
        t_proj_value_w = self.t_value_conv_w(template_map).view(m_batchsize, -1, t_w)

        t_out_w = torch.bmm(t_proj_value_w, t_attention_w.permute(0, 2, 1))
        t_out_w = t_out_w.view(m_batchsize, C, t_h, t_w)
        t_out_w = self.alpha * t_out_w + template_map



        s_proj_query_w = self.s_query_conv_w(scene_map).view(m_batchsize, -1, s_w).permute(0, 2, 1)
        s_proj_key_w = self.s_key_conv_w(scene_map).view(m_batchsize, -1, s_w)
        s_energy_w = torch.bmm(s_proj_query_w, s_proj_key_w)
        s_attention_w = self.softmax(s_energy_w)
        s_proj_value_w = self.s_value_conv_w(scene_map).view(m_batchsize, -1, s_w)

        s_out_w = torch.bmm(s_proj_value_w, s_attention_w.permute(0, 2, 1))
        s_out_w = s_out_w.view(m_batchsize, C, s_h, s_w)
        s_out_w = self.beta * s_out_w + scene_map


        t_proj_query_h = self.t_query_conv_h(template_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h).permute(0, 2, 1)
        t_proj_key_h = self.t_key_conv_h(template_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)
        t_energy_h = torch.bmm(t_proj_query_h, t_proj_key_h)
        t_attention_h = self.softmax(t_energy_h)
        t_proj_value_h = self.t_value_conv_h(template_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)

        t_out_h = torch.bmm(t_proj_value_h, t_attention_h.permute(0, 2, 1))
        t_out_h = t_out_h.view(m_batchsize, C, t_w, t_h).permute(0, 1, 3, 2)
        t_out_h = self.gamma * t_out_h + template_map

        s_proj_query_h = self.s_query_conv_h(scene_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h).permute(0, 2, 1)
        s_proj_key_h = self.s_key_conv_h(scene_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)
        s_energy_h = torch.bmm(s_proj_query_h, s_proj_key_h)
        s_attention_h = self.softmax(s_energy_h)
        s_proj_value_h = self.s_value_conv_h(scene_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)

        s_out_h = torch.bmm(s_proj_value_h, s_attention_h.permute(0, 2, 1))
        s_out_h = s_out_h.view(m_batchsize, C, s_w, s_h).permute(0, 1, 3, 2)
        s_out_h = self.omega * s_out_h + scene_map

        fused_template_map = t_out_w + t_out_h
        fused_scene_map = s_out_w + s_out_h


        return fused_template_map, fused_scene_map


class FPAM_Module(nn.Module):

    def __init__(self, in_dim=64):
        super(FPAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.t_query_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.t_query_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_w = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, rgb_t_map, rgb_s_map, x_t_map, x_s_map):

        m_batchsize, C, t_h, t_w = rgb_t_map.size()
        _, _, s_h, s_w = rgb_s_map.size()

        t_proj_query_w = self.t_query_conv_w(rgb_t_map).view(m_batchsize, -1, t_w).permute(0, 2, 1)
        t_proj_key_w = self.t_key_conv_w(rgb_t_map).view(m_batchsize, -1, t_w)
        t_energy_w = torch.bmm(t_proj_query_w, t_proj_key_w)
        t_attention_w = self.softmax(t_energy_w)
        t_proj_value_w = self.t_value_conv_w(x_t_map).view(m_batchsize, -1, t_w)

        t_out_w = torch.bmm(t_proj_value_w, t_attention_w.permute(0, 2, 1))
        t_out_w = t_out_w.view(m_batchsize, C, t_h, t_w)
        t_out_w = self.alpha * t_out_w + x_t_map



        s_proj_query_w = self.s_query_conv_w(rgb_s_map).view(m_batchsize, -1, s_w).permute(0, 2, 1)
        s_proj_key_w = self.s_key_conv_w(rgb_s_map).view(m_batchsize, -1, s_w)
        s_energy_w = torch.bmm(s_proj_query_w, s_proj_key_w)
        s_attention_w = self.softmax(s_energy_w)
        s_proj_value_w = self.s_value_conv_w(x_s_map).view(m_batchsize, -1, s_w)

        s_out_w = torch.bmm(s_proj_value_w, s_attention_w.permute(0, 2, 1))
        s_out_w = s_out_w.view(m_batchsize, C, s_h, s_w)
        s_out_w = self.beta * s_out_w + x_s_map


        t_proj_query_h = self.t_query_conv_h(rgb_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h).permute(0, 2, 1)
        t_proj_key_h = self.t_key_conv_h(rgb_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)
        t_energy_h = torch.bmm(t_proj_query_h, t_proj_key_h)
        t_attention_h = self.softmax(t_energy_h)
        t_proj_value_h = self.t_value_conv_h(x_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)

        t_out_h = torch.bmm(t_proj_value_h, t_attention_h.permute(0, 2, 1))
        t_out_h = t_out_h.view(m_batchsize, C, t_w, t_h).permute(0, 1, 3, 2)
        t_out_h = self.gamma * t_out_h + x_t_map

        s_proj_query_h = self.s_query_conv_h(rgb_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h).permute(0, 2, 1)
        s_proj_key_h = self.s_key_conv_h(rgb_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)
        s_energy_h = torch.bmm(s_proj_query_h, s_proj_key_h)
        s_attention_h = self.softmax(s_energy_h)
        s_proj_value_h = self.s_value_conv_h(x_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)

        s_out_h = torch.bmm(s_proj_value_h, s_attention_h.permute(0, 2, 1))
        s_out_h = s_out_h.view(m_batchsize, C, s_w, s_h).permute(0, 1, 3, 2)
        s_out_h = self.omega * s_out_h + x_s_map

        fused_template_map = t_out_w + t_out_h
        fused_scene_map = s_out_w + s_out_h


        return fused_template_map, fused_scene_map

class FCAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim=64):
        super(FCAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.t_query_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb_t_map, rgb_s_map, z_t_map, z_s_map):
        m_batchsize, C, t_h, t_w = rgb_t_map.size()
        _, _, s_h, s_w = rgb_s_map.size()

        t_proj_query_c = self.t_query_conv_c(rgb_t_map).view(m_batchsize, C, -1)
        t_proj_key_c = self.t_key_conv_c(rgb_t_map).view(m_batchsize, C, -1).permute(0, 2, 1)
        t_energy_c = torch.bmm(t_proj_query_c, t_proj_key_c)
        t_energy_c_new = torch.max(t_energy_c, -1, keepdim=True)[0].expand_as(t_energy_c) - t_energy_c
        t_attention_c = self.softmax(t_energy_c_new)
        t_proj_value_c = self.t_value_conv_c(z_t_map).view(m_batchsize, C, -1)

        t_out_c = torch.bmm(t_attention_c, t_proj_value_c)
        t_out_c = t_out_c.view(m_batchsize, C, t_h, t_w)
        t_out_c = self.alpha * t_out_c + z_t_map


        s_proj_query_c = self.s_query_conv_c(rgb_s_map).view(m_batchsize, C, -1)
        s_proj_key_c = self.s_key_conv_c(rgb_s_map).view(m_batchsize, C, -1).permute(0, 2, 1)
        s_energy_c = torch.bmm(s_proj_query_c, s_proj_key_c)
        s_energy_c_new = torch.max(s_energy_c, -1, keepdim=True)[0].expand_as(s_energy_c) - s_energy_c
        s_attention_c = self.softmax(s_energy_c_new)
        s_proj_value_c = self.s_value_conv_c(z_s_map).view(m_batchsize, C, -1)

        s_out_c = torch.bmm(s_attention_c, s_proj_value_c)
        s_out_c = s_out_c.view(m_batchsize, C, s_h, s_w)
        s_out_c = self.beta * s_out_c + z_s_map

        # t_out_x = torch.bmm(s_attention_c, t_proj_value_c)
        # t_out_x = t_out_x.view(m_batchsize, C, t_h, t_w)
        # t_out_x = self.gamma * t_out_x + z_t_map
        #
        # s_out_x = torch.bmm(t_attention_c, s_proj_value_c)
        # s_out_x = s_out_x.view(m_batchsize, C, s_h, s_w)
        # s_out_x = self.omega * s_out_x + z_s_map

        fused_template_map = t_out_c
        fused_scene_map = s_out_c

        return fused_template_map, fused_scene_map

class YFCAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim=64):
        super(YFCAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.t_query_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb_t_map, y_t_map):
        m_batchsize, C, t_h, t_w = rgb_t_map.size()

        t_proj_query_c = self.t_query_conv_c(rgb_t_map).view(m_batchsize, C, -1)
        t_proj_key_c = self.t_key_conv_c(rgb_t_map).view(m_batchsize, C, -1).permute(0, 2, 1)
        t_energy_c = torch.bmm(t_proj_query_c, t_proj_key_c)
        t_energy_c_new = torch.max(t_energy_c, -1, keepdim=True)[0].expand_as(t_energy_c) - t_energy_c
        t_attention_c = self.softmax(t_energy_c_new)
        t_proj_value_c = self.t_value_conv_c(y_t_map).view(m_batchsize, C, -1)

        t_out_c = torch.bmm(t_attention_c, t_proj_value_c)
        t_out_c = t_out_c.view(m_batchsize, C, t_h, t_w)
        t_out_c = self.alpha * t_out_c + y_t_map

        fused_template_map = t_out_c

        return fused_template_map


class CrossCAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CrossCAM_Module, self).__init__()

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1))

    def forward(self, template_map, roi_map):
        m_batchsize, C, t_h, t_w = template_map.size()
        _, _, r_h, r_w = roi_map.size()

        t_proj_query_c = template_map.view(m_batchsize, C, -1)
        t_proj_key_c = template_map.view(m_batchsize, C, -1).permute(0, 2, 1)
        t_energy_c = torch.bmm(t_proj_query_c, t_proj_key_c)
        t_energy_c_new = torch.max(t_energy_c, -1, keepdim=True)[0].expand_as(t_energy_c) - t_energy_c
        t_attention_c = self.softmax(t_energy_c_new)
        t_proj_value_c = template_map.view(m_batchsize, C, -1)

        r_proj_query_c = roi_map.view(m_batchsize, C, -1)
        r_proj_key_c = roi_map.view(m_batchsize, C, -1).permute(0, 2, 1)
        r_energy_c = torch.bmm(r_proj_query_c, r_proj_key_c)
        r_energy_c_new = torch.max(r_energy_c, -1, keepdim=True)[0].expand_as(r_energy_c) - r_energy_c
        r_attention_c = self.softmax(r_energy_c_new)
        r_proj_value_c = roi_map.view(m_batchsize, C, -1)

        t_out_x = torch.bmm(r_attention_c, t_proj_value_c)
        t_out_x = t_out_x.view(m_batchsize, C, t_h, t_w)
        t_out_x = self.gamma * t_out_x + template_map

        r_out_x = torch.bmm(t_attention_c, r_proj_value_c)
        r_out_x = r_out_x.view(m_batchsize, C, r_h, r_w)
        r_out_x = self.omega * r_out_x + roi_map

        fused_map = self.cnn(torch.cat([t_out_x, r_out_x], dim=1))

        return fused_map


class Dy_FCAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim=64):
        super(Dy_FCAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.t_query_conv_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.t_query_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.energy_fusion = nn.Linear(in_dim * in_dim * 4, in_dim * in_dim)

        # self.alpha = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        # self.beta = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        # self.gamma = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        # self.omega = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pc_t_map, pc_s_map, rgb_t_map, rgb_s_map):
        m_batchsize, C, t_h, t_w = rgb_t_map.size()
        _, _, s_h, s_w = rgb_s_map.size()

        t_proj_query_pc = self.t_query_conv_pc(pc_t_map).view(m_batchsize, C, -1)
        t_proj_key_pc = self.t_key_conv_pc(pc_t_map).view(m_batchsize, C, -1).permute(0, 2, 1)
        t_energy_pc = torch.bmm(t_proj_query_pc, t_proj_key_pc)
        t_energy_pc_new = torch.max(t_energy_pc, -1, keepdim=True)[0].expand_as(t_energy_pc) - t_energy_pc

        t_proj_query_rgb = self.t_query_conv_rgb(rgb_t_map).view(m_batchsize, C, -1)
        t_proj_key_rgb = self.t_key_conv_rgb(rgb_t_map).view(m_batchsize, C, -1).permute(0, 2, 1)
        t_energy_rgb = torch.bmm(t_proj_query_rgb, t_proj_key_rgb)
        t_energy_rgb_new = torch.max(t_energy_rgb, -1, keepdim=True)[0].expand_as(t_energy_rgb) - t_energy_rgb

        s_proj_query_pc = self.s_query_conv_pc(pc_s_map).view(m_batchsize, C, -1)
        s_proj_key_pc = self.s_key_conv_pc(pc_s_map).view(m_batchsize, C, -1).permute(0, 2, 1)
        s_energy_pc = torch.bmm(s_proj_query_pc, s_proj_key_pc)
        s_energy_pc_new = torch.max(s_energy_pc, -1, keepdim=True)[0].expand_as(s_energy_pc) - s_energy_pc

        s_proj_query_rgb = self.s_query_conv_rgb(rgb_s_map).view(m_batchsize, C, -1)
        s_proj_key_rgb = self.s_key_conv_rgb(rgb_s_map).view(m_batchsize, C, -1).permute(0, 2, 1)
        s_energy_rgb = torch.bmm(s_proj_query_rgb, s_proj_key_rgb)
        s_energy_rgb_new = torch.max(s_energy_rgb, -1, keepdim=True)[0].expand_as(s_energy_rgb) - s_energy_rgb

        energy_fusion = self.energy_fusion(torch.cat([t_energy_pc_new.reshape(m_batchsize, -1), t_energy_rgb_new.reshape(m_batchsize, -1),
                                                        s_energy_pc_new.reshape(m_batchsize, -1), s_energy_rgb_new.reshape(m_batchsize, -1)], dim=1))


        attention_c = self.softmax(energy_fusion.reshape(m_batchsize, C, C))

        t_proj_value_pc = self.t_value_conv_pc(pc_t_map).view(m_batchsize, C, -1)
        t_proj_value_rgb = self.t_value_conv_rgb(rgb_t_map).view(m_batchsize, C, -1)

        t_out_pc = torch.bmm(attention_c, t_proj_value_pc)
        t_out_pc = t_out_pc.view(m_batchsize, C, t_h, t_w)
        t_out_rgb = torch.bmm(attention_c, t_proj_value_rgb)
        t_out_rgb = t_out_rgb.view(m_batchsize, C, t_h, t_w)
        # a = F.avg_pool2d(self.t_value_conv_pc(pc_t_map), (40, 40)).reshape(m_batchsize, -1)
        # b = F.avg_pool2d(self.t_value_conv_rgb(rgb_t_map), (40, 40)).reshape(m_batchsize, -1)
        # a = self.alpha(a).unsqueeze(2).unsqueeze(3)
        # b = self.beta(b).unsqueeze(2).unsqueeze(3)
        #
        # t_out_c = a * t_out_pc + b * t_out_rgb

        s_proj_value_pc = self.s_value_conv_pc(pc_s_map).view(m_batchsize, C, -1)
        s_proj_value_rgb = self.s_value_conv_rgb(rgb_s_map).view(m_batchsize, C, -1)
        s_out_pc = torch.bmm(attention_c, s_proj_value_pc)
        s_out_pc = s_out_pc.view(m_batchsize, C, s_h, s_w)
        s_out_rgb = torch.bmm(attention_c, s_proj_value_rgb)
        s_out_rgb = s_out_rgb.view(m_batchsize, C, s_h, s_w)
        # g = F.avg_pool2d(self.s_value_conv_pc(pc_s_map), (64, 64)).reshape(m_batchsize, -1)
        # o = F.avg_pool2d(self.s_value_conv_rgb(rgb_s_map), (64, 64)).reshape(m_batchsize, -1)
        # g = self.gamma(g).unsqueeze(2).unsqueeze(3)
        # o = self.omega(o).unsqueeze(2).unsqueeze(3)
        #
        # s_out_c = g * s_out_pc + o * s_out_rgb

        return t_out_pc, t_out_rgb, s_out_pc, s_out_rgb


class Dy_FPAM_Module(nn.Module):

    def __init__(self, in_dim=64):
        super(Dy_FPAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.t_query_conv_h_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_h_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_h_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_h_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_h_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_h_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.t_query_conv_w_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_w_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_w_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_w_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_w_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_w_pc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.t_query_conv_h_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_h_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_h_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_h_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_h_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_h_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.t_query_conv_w_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_key_conv_w_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.t_value_conv_w_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv_w_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_key_conv_w_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.s_value_conv_w_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.t_energy_fusion_w = nn.Linear(40 * 40 * 2, 40 * 40)
        self.s_energy_fusion_w = nn.Linear(64 * 64 * 2, 64 * 64)
        self.t_energy_fusion_h = nn.Linear(40 * 40 * 2, 40 * 40)
        self.s_energy_fusion_h = nn.Linear(64 * 64 * 2, 64 * 64)


        self.a = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        self.b = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        self.c = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        self.d = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())

        self.e = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        self.f = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        self.g = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())
        self.h = nn.Sequential(nn.Linear(64, 1, bias=False), nn.Sigmoid())

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, rgb_t_map, rgb_s_map, x_t_map, x_s_map):

        m_batchsize, C, t_h, t_w = rgb_t_map.size()
        _, _, s_h, s_w = rgb_s_map.size()

        t_proj_query_w_pc = self.t_query_conv_w_pc(x_t_map).view(m_batchsize, -1, t_w).permute(0, 2, 1)
        t_proj_key_w_pc = self.t_key_conv_w_pc(x_t_map).view(m_batchsize, -1, t_w)
        t_energy_w_pc = torch.bmm(t_proj_query_w_pc, t_proj_key_w_pc)

        t_proj_query_w_rgb = self.t_query_conv_w_rgb(rgb_t_map).view(m_batchsize, -1, t_w).permute(0, 2, 1)
        t_proj_key_w_rgb = self.t_key_conv_w_rgb(rgb_t_map).view(m_batchsize, -1, t_w)
        t_energy_w_rgb = torch.bmm(t_proj_query_w_rgb, t_proj_key_w_rgb)

        t_energy_w_fusion = self.t_energy_fusion_w(
            torch.cat([t_energy_w_pc.reshape(m_batchsize, -1), t_energy_w_rgb.reshape(m_batchsize, -1)], dim=1))

        t_attention_w = self.softmax(t_energy_w_fusion.reshape(m_batchsize, t_w, t_w))
        t_proj_value_w_pc = self.t_value_conv_w_pc(x_t_map).view(m_batchsize, -1, t_w)
        t_proj_value_w_rgb = self.t_value_conv_w_rgb(rgb_t_map).view(m_batchsize, -1, t_w)

        t_out_w_pc = torch.bmm(t_proj_value_w_pc, t_attention_w)
        t_out_w_pc = t_out_w_pc.view(m_batchsize, C, t_h, t_w)
        t_out_w_rgb = torch.bmm(t_proj_value_w_rgb, t_attention_w)
        t_out_w_rgb = t_out_w_rgb.view(m_batchsize, C, t_h, t_w)
        pc_t_avg = F.avg_pool2d(self.t_value_conv_w_pc(x_t_map), (40, 40)).reshape(m_batchsize, -1)
        rgb_t_avg = F.avg_pool2d(self.t_value_conv_w_rgb(rgb_t_map), (40, 40)).reshape(m_batchsize, -1)
        a = self.a(pc_t_avg).unsqueeze(2).unsqueeze(3)
        b = self.b(rgb_t_avg).unsqueeze(2).unsqueeze(3)
        t_out_w = a * t_out_w_pc + b * t_out_w_rgb + x_t_map

        t_proj_query_h_pc = self.t_query_conv_h_pc(x_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h).permute(0, 2, 1)
        t_proj_key_h_pc = self.t_key_conv_h_pc(x_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)
        t_energy_h_pc = torch.bmm(t_proj_query_h_pc, t_proj_key_h_pc)

        t_proj_query_h_rgb = self.t_query_conv_h_rgb(rgb_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h).permute(0, 2, 1)
        t_proj_key_h_rgb = self.t_key_conv_h_rgb(rgb_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)
        t_energy_h_rgb = torch.bmm(t_proj_query_h_rgb, t_proj_key_h_rgb)

        t_energy_h_fusion = self.t_energy_fusion_h(
            torch.cat([t_energy_h_pc.reshape(m_batchsize, -1), t_energy_h_rgb.reshape(m_batchsize, -1)], dim=1))
        t_attention_h = self.softmax(t_energy_h_fusion.reshape(m_batchsize, t_h, t_h))

        t_proj_value_h_pc = self.t_value_conv_h_pc(x_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)
        t_proj_value_h_rgb = self.t_value_conv_h_rgb(rgb_t_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)

        t_out_h_pc = torch.bmm(t_proj_value_h_pc, t_attention_h.permute(0, 2, 1))
        t_out_h_pc = t_out_h_pc.view(m_batchsize, C, t_h, t_w).permute(0, 1, 3, 2)
        t_out_h_rgb = torch.bmm(t_proj_value_h_rgb, t_attention_h.permute(0, 2, 1))
        t_out_h_rgb = t_out_h_rgb.view(m_batchsize, C, t_h, t_w).permute(0, 1, 3, 2)
        c = self.c(pc_t_avg).unsqueeze(2).unsqueeze(3)
        d = self.d(rgb_t_avg).unsqueeze(2).unsqueeze(3)
        t_out_h = c * t_out_h_pc + d * t_out_h_rgb + x_t_map

        s_proj_query_w_pc = self.s_query_conv_w_pc(x_s_map).view(m_batchsize, -1, s_w).permute(0, 2, 1)
        s_proj_key_w_pc = self.s_key_conv_w_pc(x_s_map).view(m_batchsize, -1, s_w)
        s_energy_w_pc = torch.bmm(s_proj_query_w_pc, s_proj_key_w_pc)

        s_proj_query_w_rgb = self.s_query_conv_w_rgb(rgb_s_map).view(m_batchsize, -1, s_w).permute(0, 2, 1)
        s_proj_key_w_rgb = self.s_key_conv_w_rgb(rgb_s_map).view(m_batchsize, -1, s_w)
        s_energy_w_rgb = torch.bmm(s_proj_query_w_rgb, s_proj_key_w_rgb)

        s_energy_w_fusion = self.s_energy_fusion_w(
            torch.cat([s_energy_w_pc.reshape(m_batchsize, -1), s_energy_w_rgb.reshape(m_batchsize, -1)], dim=1))

        s_attention_w = self.softmax(s_energy_w_fusion.reshape(m_batchsize, s_w, s_w))
        s_proj_value_w_pc = self.s_value_conv_w_pc(x_s_map).view(m_batchsize, -1, s_w)
        s_proj_value_w_rgb = self.s_value_conv_w_rgb(rgb_s_map).view(m_batchsize, -1, s_w)

        s_out_w_pc = torch.bmm(s_proj_value_w_pc, s_attention_w)
        s_out_w_pc = s_out_w_pc.view(m_batchsize, C, s_h, s_w)
        s_out_w_rgb = torch.bmm(s_proj_value_w_rgb, s_attention_w)
        s_out_w_rgb = s_out_w_rgb.view(m_batchsize, C, s_h, s_w)
        pc_s_avg = F.avg_pool2d(self.s_value_conv_w_pc(x_s_map), (64, 64)).reshape(m_batchsize, -1)
        rgb_s_avg = F.avg_pool2d(self.s_value_conv_w_rgb(rgb_s_map), (64, 64)).reshape(m_batchsize, -1)
        e = self.e(pc_s_avg).unsqueeze(2).unsqueeze(3)
        f = self.f(rgb_s_avg).unsqueeze(2).unsqueeze(3)
        s_out_w = e * s_out_w_pc + f * s_out_w_rgb + x_s_map

        s_proj_query_h_pc = self.s_query_conv_h_pc(x_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1,                                                                                          s_h).permute(0, 2, 1)
        s_proj_key_h_pc = self.s_key_conv_h_pc(x_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)
        s_energy_h_pc = torch.bmm(s_proj_query_h_pc, s_proj_key_h_pc)

        s_proj_query_h_rgb = self.s_query_conv_h_rgb(rgb_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1,
                                                                                                      s_h).permute(0, 2,
                                                                                                                   1)
        s_proj_key_h_rgb = self.s_key_conv_h_rgb(rgb_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)
        s_energy_h_rgb = torch.bmm(s_proj_query_h_rgb, s_proj_key_h_rgb)

        s_energy_h_fusion = self.s_energy_fusion_h(
            torch.cat([s_energy_h_pc.reshape(m_batchsize, -1), s_energy_h_rgb.reshape(m_batchsize, -1)], dim=1))
        s_attention_h = self.softmax(s_energy_h_fusion.reshape(m_batchsize, s_h, s_h))

        s_proj_value_h_pc = self.s_value_conv_h_pc(x_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)
        s_proj_value_h_rgb = self.s_value_conv_h_rgb(rgb_s_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1,
                                                                                                      s_h)

        s_out_h_pc = torch.bmm(s_proj_value_h_pc, s_attention_h.permute(0, 2, 1))
        s_out_h_pc = s_out_h_pc.view(m_batchsize, C, s_h, s_w).permute(0, 1, 3, 2)
        s_out_h_rgb = torch.bmm(s_proj_value_h_rgb, s_attention_h.permute(0, 2, 1))
        s_out_h_rgb = s_out_h_rgb.view(m_batchsize, C, s_h, s_w).permute(0, 1, 3, 2)
        g = self.g(pc_s_avg).unsqueeze(2).unsqueeze(3)
        h = self.h(rgb_s_avg).unsqueeze(2).unsqueeze(3)
        s_out_h = g * s_out_h_pc + h * s_out_h_rgb + x_s_map


        fused_template_map = t_out_w + t_out_h
        fused_scene_map = s_out_w + s_out_h


        return fused_template_map, fused_scene_map