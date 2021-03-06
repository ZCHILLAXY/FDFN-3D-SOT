import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self,
                 num_point_features=3,
                 num_filters=[64],
                 template=False):
        super().__init__()
        self.use_norm = True
        self.with_distance = True
        self.use_absolute_xyz = True
        self.with_bias = True
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        if self.with_bias:
            num_point_features += 3

        self.num_filters = num_filters
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = cfg.VOXEL_X_SIZE
        self.voxel_y = cfg.VOXEL_Y_SIZE
        self.voxel_z = cfg.VOXEL_Z_SIZE
        if template:
            x_min = cfg.TEMPLATE_X_MIN
            y_min = cfg.TEMPLATE_Y_MIN
            z_min = cfg.TEMPLATE_Z_MIN
        else:
            x_min = cfg.SCENE_X_MIN
            y_min = cfg.SCENE_Y_MIN
            z_min = cfg.SCENE_Z_MIN
        self.x_offset = self.voxel_x / 2 + x_min
        self.y_offset = self.voxel_y / 2 + y_min
        self.z_offset = self.voxel_z / 2 + z_min


    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, x_0, x_1, x_2, box):
        voxel_features, voxel_num_points, coords = x_0, x_1, x_2
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            object_center = box[:, 0:3].unsqueeze(1)
            points_to_center = voxel_features[:, :, :3] - object_center
            points_distance = torch.norm(points_to_center, 2, 2, keepdim=True)
            # points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_distance)

        if self.with_bias:
            object_center = box[:, 0:3].unsqueeze(1)
            object_range = box[:, 3:6].unsqueeze(1).type_as(voxel_features).view(-1, 1, 3)
            points_bias = (voxel_features[:, :, :3] - object_center) / object_range
            features.append(points_bias)

        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        if features.shape[0] > 0:
            for pfn in self.pfn_layers:
                features = pfn(features)
            features = features.squeeze()
        else:
            features = torch.zeros([features.shape[0], 64]).cuda()
        return features
