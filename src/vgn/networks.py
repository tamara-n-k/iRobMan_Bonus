import torch

from vgn.ConvONets.conv_onet.config import get_model


def get_network(name):
    models = {
        "giga_aff": GIGAAff,
        "giga": GIGA,
        "giga_geo": GIGAGeo,
        "giga_detach": GIGADetach,
    }
    try:
        return models[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported model type: {name}") from exc


def load_network(path, device, model_type=None):
    if model_type is None:
        model_name = "_".join(path.stem.split("_")[1:-1])
    else:
        model_name = model_type
    print(f"Loading [{model_name}] model from {path}")
    net = get_network(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def GIGAAff():
    config = {
        "encoder": "voxel_simple_local",
        "encoder_kwargs": {
            "plane_type": ["xz", "xy", "yz"],
            "plane_resolution": 40,
            "unet": True,
            "unet_kwargs": {
                "depth": 3,
                "merge_mode": "concat",
                "start_filts": 32,
            },
        },
        "decoder": "simple_local",
        "decoder_tsdf": False,
        "decoder_kwargs": {
            "dim": 3,
            "sample_mode": "bilinear",
            "hidden_size": 32,
            "concat_feat": True,
        },
        "padding": 0,
        "c_dim": 32,
    }
    return get_model(config)


def GIGA():
    config = {
        "encoder": "voxel_simple_local",
        "encoder_kwargs": {
            "plane_type": ["xz", "xy", "yz"],
            "plane_resolution": 40,
            "unet": True,
            "unet_kwargs": {
                "depth": 3,
                "merge_mode": "concat",
                "start_filts": 32,
            },
        },
        "decoder": "simple_local",
        "decoder_tsdf": True,
        "decoder_kwargs": {
            "dim": 3,
            "sample_mode": "bilinear",
            "hidden_size": 32,
            "concat_feat": True,
        },
        "padding": 0,
        "c_dim": 32,
    }
    return get_model(config)


def GIGAGeo():
    config = {
        "encoder": "voxel_simple_local",
        "encoder_kwargs": {
            "plane_type": ["xz", "xy", "yz"],
            "plane_resolution": 40,
            "unet": True,
            "unet_kwargs": {
                "depth": 3,
                "merge_mode": "concat",
                "start_filts": 32,
            },
        },
        "decoder": "simple_local",
        "decoder_tsdf": True,
        "tsdf_only": True,
        "decoder_kwargs": {
            "dim": 3,
            "sample_mode": "bilinear",
            "hidden_size": 32,
            "concat_feat": True,
        },
        "padding": 0,
        "c_dim": 32,
    }
    return get_model(config)


def GIGADetach():
    config = {
        "encoder": "voxel_simple_local",
        "encoder_kwargs": {
            "plane_type": ["xz", "xy", "yz"],
            "plane_resolution": 40,
            "unet": True,
            "unet_kwargs": {
                "depth": 3,
                "merge_mode": "concat",
                "start_filts": 32,
            },
        },
        "decoder": "simple_local",
        "decoder_tsdf": True,
        "detach_tsdf": True,
        "decoder_kwargs": {
            "dim": 3,
            "sample_mode": "bilinear",
            "hidden_size": 32,
            "concat_feat": True,
        },
        "padding": 0,
        "c_dim": 32,
    }
    return get_model(config)
