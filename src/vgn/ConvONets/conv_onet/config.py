from torch import nn

from vgn.ConvONets.conv_onet import models
from vgn.ConvONets.encoder import encoder_dict


def get_model(cfg, device=None, dataset=None, **kwargs):
    del dataset, kwargs
    decoder = cfg["decoder"]
    encoder = cfg["encoder"]
    c_dim = cfg["c_dim"]
    decoder_kwargs = cfg["decoder_kwargs"]
    encoder_kwargs = cfg["encoder_kwargs"]
    padding = cfg["padding"]
    if padding is None:
        padding = 0.1

    tsdf_only = cfg.get("tsdf_only", False)
    detach_tsdf = cfg.get("detach_tsdf", False)

    if tsdf_only:
        decoders = []
    else:
        decoder_qual = models.decoder_dict[decoder](
            c_dim=c_dim,
            padding=padding,
            out_dim=1,
            **decoder_kwargs,
        )
        decoder_rot = models.decoder_dict[decoder](
            c_dim=c_dim,
            padding=padding,
            out_dim=4,
            **decoder_kwargs,
        )
        decoder_width = models.decoder_dict[decoder](
            c_dim=c_dim,
            padding=padding,
            out_dim=1,
            **decoder_kwargs,
        )
        decoders = [decoder_qual, decoder_rot, decoder_width]

    if cfg["decoder_tsdf"] or tsdf_only:
        decoder_tsdf = models.decoder_dict[decoder](
            c_dim=c_dim,
            padding=padding,
            out_dim=1,
            **decoder_kwargs,
        )
        decoders.append(decoder_tsdf)

    if encoder == "idx":
        if dataset is None:
            raise ValueError("dataset is required when encoder='idx'")
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim,
            padding=padding,
            **encoder_kwargs,
        )
    else:
        encoder = None

    if tsdf_only:
        return models.ConvolutionalOccupancyNetworkGeometry(
            decoder_tsdf,
            encoder,
            device=device,
        )
    return models.ConvolutionalOccupancyNetwork(
        decoders,
        encoder,
        device=device,
        detach_tsdf=detach_tsdf,
    )
