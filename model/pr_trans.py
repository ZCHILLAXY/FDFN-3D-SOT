import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from layers import ProgressiveSample
from .transformer_block import TransformerEncoderLayer


def conv3x3(in_planes,
            out_planes,
            stride=1,
            groups=1,
            dilation=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False)


def conv1x1(in_planes,
            out_planes,
            stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BottleneckLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels):
        super().__init__()
        self.conv1 = conv1x1(in_channels,
                             inter_channels)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = conv3x3(inter_channels,
                             inter_channels)
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.conv3 = conv1x1(inter_channels,
                             out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(conv1x1(in_channels, out_channels),
                                            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PSViTLayer(nn.Module):
    def __init__(self,
                 feat_size,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 position_layer=None,
                 pred_offset=True,
                 gamma=0.1,
                 offset_bias=False):
        super().__init__()

        self.feat_size = float(feat_size)

        self.transformer_layer = TransformerEncoderLayer(dim,
                                                         num_heads,
                                                         mlp_ratio,
                                                         qkv_bias,
                                                         qk_scale,
                                                         drop,
                                                         attn_drop,
                                                         drop_path,
                                                         act_layer,
                                                         norm_layer)
        self.sampler = ProgressiveSample(gamma)

        self.position_layer = position_layer
        if self.position_layer is None:
            self.position_layer = nn.Linear(2, dim)

        self.offset_layer = None
        if pred_offset:
            self.offset_layer = nn.Linear(dim, 2, bias=offset_bias)

    def reset_offset_weight(self):
        if self.offset_layer is None:
            return
        nn.init.constant_(self.offset_layer.weight, 0)
        if self.offset_layer.bias is not None:
            nn.init.constant_(self.offset_layer.bias, 0)

    def forward(self,
                x,
                point,
                offset=None,
                pre_out=None):
        """
        :param x: [n, dim, h, w]
        :param point: [n, point_num, 2]
        :param offset: [n, point_num, 2]
        :param pre_out: [n, point_num, dim]
        """
        if offset is None:
            offset = torch.zeros_like(point)

        sample_feat = self.sampler(x, point, offset)
        sample_point = point + offset.detach()

        pos_feat = self.position_layer(sample_point / self.feat_size)

        attn_feat = sample_feat + pos_feat
        if pre_out is not None:
            attn_feat = attn_feat + pre_out

        attn_feat = self.transformer_layer(attn_feat)

        out_offset = None
        if self.offset_layer is not None:
            out_offset = self.offset_layer(attn_feat)

        return attn_feat, out_offset, sample_point


class PSViT(nn.Module):
    def __init__(self,
                 in_chans=64,
                 img_size=64,
                 num_point_w=13,
                 num_point_h=13,
                 downsample_ratio=1,
                 num_iters=4,
                 depth=4,
                 embed_dim=256,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 offset_gamma=0.1,
                 offset_bias=False,
                 stem_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        assert num_iters >= 1

        self.img_size = img_size
        self.feat_size = img_size // downsample_ratio

        self.num_point_w = num_point_w
        self.num_point_h = num_point_h

        self.register_buffer('point_coord', self._get_initial_point())

        self.pos_layer = nn.Linear(2, self.embed_dim)

        self.stem = stem_layer
        if self.stem is None:
            self.stem = nn.Sequential(nn.Conv2d(in_chans,
                                                64,
                                                kernel_size=7,
                                                padding=3,
                                                stride=2,
                                                bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3,
                                                   stride=2,
                                                   padding=1),
                                      BottleneckLayer(in_chans, 128, self.embed_dim),
                                      BottleneckLayer(self.embed_dim, 128, self.embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.ps_layers = nn.ModuleList()
        for i in range(num_iters):
            self.ps_layers.append(PSViTLayer(feat_size=self.feat_size,
                                             dim=self.embed_dim,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop_rate,
                                             attn_drop=attn_drop_rate,
                                             drop_path=dpr[i],
                                             norm_layer=norm_layer,
                                             position_layer=self.pos_layer,
                                             pred_offset=i < num_iters - 1,
                                             gamma=offset_gamma,
                                             offset_bias=offset_bias))

        self.trans_layers = nn.ModuleList()
        trans_depth = depth - num_iters
        for i in range(trans_depth):
            self.trans_layers.append(TransformerEncoderLayer(dim=self.embed_dim,
                                                             num_heads=num_heads,
                                                             mlp_ratio=mlp_ratio,
                                                             qkv_bias=qkv_bias,
                                                             qk_scale=qk_scale,
                                                             drop=drop_rate,
                                                             attn_drop=attn_drop_rate,
                                                             drop_path=dpr[i +
                                                                           num_iters],
                                                             norm_layer=norm_layer))

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        for layer in self.ps_layers:
            layer.reset_offset_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_initial_point(self):
        patch_size_w = self.feat_size / self.num_point_w
        patch_size_h = self.feat_size / self.num_point_h
        coord_w = torch.Tensor(
            [i * patch_size_w for i in range(self.num_point_w)])
        coord_w += patch_size_w / 2
        coord_h = torch.Tensor(
            [i * patch_size_h for i in range(self.num_point_h)])
        coord_h += patch_size_h / 2

        grid_x, grid_y = torch.meshgrid(coord_w, coord_h)
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)
        point_coord = torch.cat([grid_y, grid_x], dim=0)
        point_coord = point_coord.view(2, -1)
        point_coord = point_coord.permute(1, 0).contiguous().unsqueeze(0)

        return point_coord

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cls_token is not None:
            return {'cls_token'}
        else:
            return {}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_feature(self, x):
        batch_size = x.size(0)
        point = self.point_coord.repeat(batch_size, 1, 1)

        x = self.stem(x)

        ps_out = None
        offset = None

        for layer in self.ps_layers:
            ps_out, offset, point = layer(x,
                                          point,
                                          offset,
                                          ps_out)

        trans_out = ps_out
        for layer in self.trans_layers:
            trans_out = layer(trans_out)

        trans_out = self.norm(trans_out).permute(0, 2, 1).reshape(batch_size, -1, self.num_point_w, self.num_point_h)

        # if self.cls_token is not None:
        #     out_feat = trans_out[:, 0]
        # else:
        #     trans_out = trans_out.permute(0, 2, 1)
        #     out_feat = self.avgpool(trans_out).view(batch_size, self.embed_dim)

        return trans_out

    def forward(self, x):
        assert x.shape[-1] == self.img_size and x.shape[-2] == self.img_size
        x = self.forward_feature(x)
        return x

# class PRSiT(nn.Module):
#     def __init__(self,
#                  img_size,
#                  num_point_w,
#                  num_point_h,
#                  in_chans=64,
#                  downsample_ratio=1,
#                  num_iters=2,
#                  depth=2,
#                  embed_dim=256,
#                  num_heads=2,
#                  mlp_ratio=2.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  norm_layer=nn.LayerNorm,
#                  stem_layer=None,
#                  offset_gamma=0.1,
#                  offset_bias=False):
#         super().__init__()
#         self.embed_dim = embed_dim
#         assert num_iters >= 1
#
#         self.img_size = img_size
#         self.feat_size = img_size // downsample_ratio
#
#         self.num_point_w = num_point_w
#         self.num_point_h = num_point_h
#
#         self.register_buffer('point_coord', self._get_initial_point())
#
#         self.pos_layer = nn.Linear(2, self.embed_dim)
#
#         self.stem = stem_layer
#         if self.stem is None:
#             self.stem_rgb = nn.Sequential(nn.Conv2d(in_chans,
#                                                 128,
#                                                 kernel_size=3,
#                                                 padding=1,
#                                                 stride=1,
#                                                 bias=False),
#                                       nn.BatchNorm2d(128),
#                                       nn.ReLU(inplace=True),
#                                       nn.MaxPool2d(kernel_size=3,
#                                                    stride=1,
#                                                    padding=1),
#                                       BottleneckLayer(128, 128, self.embed_dim),
#                                       BottleneckLayer(self.embed_dim, 128, self.embed_dim))
#             self.stem_pc = nn.Sequential(nn.Conv2d(in_chans,
#                                                 128,
#                                                 kernel_size=3,
#                                                 padding=1,
#                                                 stride=1,
#                                                 bias=False),
#                                       nn.BatchNorm2d(128),
#                                       nn.ReLU(inplace=True),
#                                       nn.MaxPool2d(kernel_size=3,
#                                                    stride=1,
#                                                    padding=1),
#                                       BottleneckLayer(128, 128, self.embed_dim),
#                                       BottleneckLayer(self.embed_dim, 128, self.embed_dim))
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.ps_layers = nn.ModuleList()
#         for i in range(num_iters):
#             self.ps_layers.append(PSViTLayer(feat_size=self.feat_size,
#                                              dim=self.embed_dim,
#                                              num_heads=num_heads,
#                                              mlp_ratio=mlp_ratio,
#                                              qkv_bias=qkv_bias,
#                                              qk_scale=qk_scale,
#                                              drop=drop_rate,
#                                              attn_drop=attn_drop_rate,
#                                              drop_path=dpr[i],
#                                              norm_layer=norm_layer,
#                                              position_layer=self.pos_layer,
#                                              pred_offset=i < num_iters - 1,
#                                              gamma=offset_gamma,
#                                              offset_bias=offset_bias))
#             self.ps_layers.append(PSViTLayer(feat_size=self.feat_size,
#                                              dim=self.embed_dim,
#                                              num_heads=num_heads,
#                                              mlp_ratio=mlp_ratio,
#                                              qkv_bias=qkv_bias,
#                                              qk_scale=qk_scale,
#                                              drop=drop_rate,
#                                              attn_drop=attn_drop_rate,
#                                              drop_path=dpr[i],
#                                              norm_layer=norm_layer,
#                                              position_layer=self.pos_layer,
#                                              pred_offset=i < num_iters - 1,
#                                              gamma=offset_gamma,
#                                              offset_bias=offset_bias))
#
#         self.trans_layers = nn.ModuleList()
#         trans_depth = depth - num_iters
#         for i in range(trans_depth):
#             self.trans_layers.append(TransformerEncoderLayer(dim=self.embed_dim,
#                                                              num_heads=num_heads,
#                                                              mlp_ratio=mlp_ratio,
#                                                              qkv_bias=qkv_bias,
#                                                              qk_scale=qk_scale,
#                                                              drop=drop_rate,
#                                                              attn_drop=attn_drop_rate,
#                                                              drop_path=dpr[i +
#                                                                            num_iters],
#                                                              norm_layer=norm_layer))
#             self.trans_layers.append(TransformerEncoderLayer(dim=self.embed_dim,
#                                                              num_heads=num_heads,
#                                                              mlp_ratio=mlp_ratio,
#                                                              qkv_bias=qkv_bias,
#                                                              qk_scale=qk_scale,
#                                                              drop=drop_rate,
#                                                              attn_drop=attn_drop_rate,
#                                                              drop_path=dpr[i +
#                                                                            num_iters],
#                                                              norm_layer=norm_layer))
#
#         self.pc_norm = norm_layer(embed_dim)
#         self.rgb_norm = norm_layer(embed_dim)
#
#         self.apply(self._init_weights)
#
#
#         for layer in self.ps_layers:
#             layer.reset_offset_weight()
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def _get_initial_point(self):
#         patch_size_w = self.feat_size / self.num_point_w
#         patch_size_h = self.feat_size / self.num_point_h
#         coord_w = torch.Tensor(
#             [i * patch_size_w for i in range(self.num_point_w)])
#         coord_w += patch_size_w / 2
#         coord_h = torch.Tensor(
#             [i * patch_size_h for i in range(self.num_point_h)])
#         coord_h += patch_size_h / 2
#
#         grid_x, grid_y = torch.meshgrid(coord_w, coord_h)
#         grid_x = grid_x.unsqueeze(0)
#         grid_y = grid_y.unsqueeze(0)
#         point_coord = torch.cat([grid_y, grid_x], dim=0)
#         point_coord = point_coord.view(2, -1)
#         point_coord = point_coord.permute(1, 0).contiguous().unsqueeze(0)
#
#         return point_coord
#
#     def forward_feature(self, x):
#         batch_size = x.size(0)
#         pc_point = self.point_coord.repeat(batch_size, 1, 1)
#         rgb_point = self.point_coord.repeat(batch_size, 1, 1)
#
#         pc = self.stem_pc(x)
#         rgb = self.stem_rgb(rgb)
#
#         pc_ps_out = None
#         pc_offset = None
#         rgb_ps_out = None
#         rgb_offset = None
#
#         for idx, layer in enumerate(self.ps_layers):
#             if idx % 2 == 0:
#                 pc_ps_out, pc_offset, pc_point = layer(pc, rgb_point, rgb_offset, rgb_ps_out)
#             else:
#                 rgb_ps_out, rgb_offset, rgb_point = layer(rgb, pc_point, pc_offset, pc_ps_out)
#
#         pc_trans_out = pc_ps_out
#         rgb_trans_out = rgb_ps_out
#
#         for idx, layer in enumerate(self.trans_layers):
#             if idx % 2 == 0:
#                 pc_trans_out = layer(pc_trans_out)
#             else:
#                 rgb_trans_out = layer(rgb_trans_out)
#
#         return pc_trans_out, rgb_trans_out
#
#
#     def forward(self, x):
#         assert x.shape[-1] == self.img_size
#         x = self.forward_feature(x)
#         out = self.pc_norm(x)
#         out = out.reshape(-1, self.embed_dim, self.num_point_h, self.num_point_w)
#         return out
