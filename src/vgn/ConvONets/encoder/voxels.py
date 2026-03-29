import torch
import torch.nn as nn
import torch.nn.functional as F

from vgn.ConvONets.common import coordinate2index, normalize_3d_coordinate, normalize_coordinate
from vgn.ConvONets.encoder.unet import UNet


def _scatter_mean(src, index, out):
    expanded_index = index.expand(-1, src.size(1), -1)
    out.scatter_add_(2, expanded_index, src)
    counts = out.new_zeros(out.shape)
    counts.scatter_add_(2, expanded_index, torch.ones_like(src))
    return out / counts.clamp_min(1)


class LocalVoxelEncoder(nn.Module):
    def __init__(
        self,
        dim=3,
        c_dim=128,
        unet=False,
        unet_kwargs=None,
        unet3d=False,
        unet3d_kwargs=None,
        plane_resolution=512,
        grid_resolution=None,
        plane_type="xz",
        kernel_size=3,
        padding=0.1,
    ):
        del dim, unet3d, unet3d_kwargs
        super().__init__()
        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1)

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        self.c_dim = c_dim
        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)

        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1)
        fea_plane = _scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(
            p.size(0),
            self.c_dim,
            self.reso_plane,
            self.reso_plane,
        )

        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        if self.reso_grid is None:
            raise ValueError("grid_resolution is required when using grid features")

        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")

        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = _scatter_mean(c, index, out=fea_grid)
        return fea_grid.reshape(
            p.size(0),
            self.c_dim,
            self.reso_grid,
            self.reso_grid,
            self.reso_grid,
        )

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        n_voxel = x.size(1) * x.size(2) * x.size(3)

        coord1 = torch.linspace(-0.5, 0.5, x.size(1), device=device)
        coord2 = torch.linspace(-0.5, 0.5, x.size(2), device=device)
        coord3 = torch.linspace(-0.5, 0.5, x.size(3), device=device)

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)
        p = torch.stack([coord1, coord2, coord3], dim=4).view(batch_size, n_voxel, -1)

        x = x.unsqueeze(1)
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1)
        c = c.permute(0, 2, 1)

        fea = {}
        if "grid" in self.plane_type:
            fea["grid"] = self.generate_grid_features(p, c)
        else:
            if "xz" in self.plane_type:
                fea["xz"] = self.generate_plane_features(p, c, plane="xz")
            if "xy" in self.plane_type:
                fea["xy"] = self.generate_plane_features(p, c, plane="xy")
            if "yz" in self.plane_type:
                fea["yz"] = self.generate_plane_features(p, c, plane="yz")

        return fea
