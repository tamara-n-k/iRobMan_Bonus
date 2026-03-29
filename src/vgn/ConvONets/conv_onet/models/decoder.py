import torch
import torch.nn as nn
import torch.nn.functional as F

from vgn.ConvONets.common import normalize_3d_coordinate, normalize_coordinate
from vgn.ConvONets.layers import ResnetBlockFC


class FCDecoder(nn.Module):
    def __init__(
        self,
        dim=3,
        c_dim=128,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
    ):
        del leaky
        super().__init__()
        self.c_dim = c_dim
        self.fc = nn.Linear(dim + c_dim, out_dim)
        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0
        return F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0
        return F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1).squeeze(-1)

    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)

        return self.fc(torch.cat((c, p), dim=2)).squeeze(-1)


class LocalDecoder(nn.Module):
    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        no_xyz=False,
    ):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)]
            )

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for _ in range(n_blocks)]
        )
        self.fc_out = nn.Linear(hidden_size, out_dim)
        self.actvn = F.relu if not leaky else lambda x: F.leaky_relu(x, 0.2)
        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0
        return F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0
        return F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1).squeeze(-1)

    def forward(self, p, c_plane, **kwargs):
        del kwargs
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c = self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)

        p = p.float()

        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size, device=p.device)
        else:
            net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        return self.fc_out(self.actvn(net)).squeeze(-1)
