import torch


def normalize_coordinate(p, padding=0.1, plane="xz"):
    if plane == "xz":
        xy = p[:, :, [0, 2]]
    elif plane == "xy":
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6)
    xy_new = xy_new + 0.5

    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0.1):
    p_nor = p / (1 + padding + 10e-4)
    p_nor = p_nor + 0.5
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def coordinate2index(x, reso, coord_type="2d"):
    x = (x * reso).long()
    if coord_type == "2d":
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == "3d":
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    else:
        raise ValueError(f"Unsupported coord_type: {coord_type}")
    return index[:, None, :]
