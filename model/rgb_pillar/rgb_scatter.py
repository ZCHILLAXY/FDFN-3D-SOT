import torch
import torch.nn as nn
from config import cfg

class RGBScatter(nn.Module):
    def __init__(self,
                 num_bev_features=64,
                 template=False):
        super().__init__()

        self.num_bev_features = num_bev_features
        if template:
            self.nx = cfg.TEMPLATE_INPUT_WIDTH
            self.ny = 1
            self.nz = cfg.TEMPLATE_INPUT_WIDTH
        else:
            self.nx = cfg.SCENE_INPUT_WIDTH
            self.ny = 1
            self.nz = cfg.SCENE_INPUT_WIDTH

    def forward(self, x_0, x_1, batchsize):
        x_0 = x_0.view(-1, 64)
        rgb_depth_features, coords = x_0, x_1
        batch_spatial_features = []
        batch_size = batchsize

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=rgb_depth_features.dtype,
                device=rgb_depth_features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = rgb_depth_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.ny, self.nx, self.nz)

        return batch_spatial_features
