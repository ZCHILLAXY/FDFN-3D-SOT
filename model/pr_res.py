import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 1)
        )

    def forward(self, x):
        return x+self.block(x)

class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 1)

        self.split_conv0 = BasicConv(out_channels, out_channels, 1)
        self.split_conv1 = BasicConv(out_channels, out_channels, 1)
        self.blocks_conv = nn.Sequential(
            Resblock(channels=out_channels, hidden_channels=out_channels // 2),
            BasicConv(out_channels, out_channels, 1)
        )
        self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)


    def forward(self, pc, rgb):
        rgb = self.downsample_conv(rgb)

        x0 = self.split_conv0(pc)
        x1 = self.split_conv1(rgb)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x

class PRFuseBlock(nn.Module):
    def __init__(self):
        super(PRFuseBlock, self).__init__()
        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.resblock = Resblock_body(self.inplanes, 64)


    def forward(self, pc, rgb):
        rgb = self.conv1(rgb)
        out = self.resblock(pc, rgb)

        return out
