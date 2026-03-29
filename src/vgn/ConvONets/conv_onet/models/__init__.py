import torch
import torch.nn as nn

from vgn.ConvONets.conv_onet.models import decoder


decoder_dict = {
    "simple_fc": decoder.FCDecoder,
    "simple_local": decoder.LocalDecoder,
}


class ConvolutionalOccupancyNetwork(nn.Module):
    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()

        self.decoder_qual = decoders[0].to(device)
        self.decoder_rot = decoders[1].to(device)
        self.decoder_width = decoders[2].to(device)
        if len(decoders) == 4:
            self.decoder_tsdf = decoders[3].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        self.detach_tsdf = detach_tsdf

    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        del sample
        if isinstance(p, dict):
            _batch_size = p["p"].size(0)
        else:
            _batch_size = p.size(0)

        c = self.encode_inputs(inputs)
        qual, rot, width = self.decode(p, c)
        if p_tsdf is not None:
            if self.detach_tsdf:
                for key, value in c.items():
                    c[key] = value.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
            return qual, rot, width, tsdf
        return qual, rot, width

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        return self.decoder_tsdf(p_tsdf, c, **kwargs)

    def encode_inputs(self, inputs):
        if self.encoder is not None:
            return self.encoder(inputs)
        return torch.empty(inputs.size(0), 0)

    def decode(self, p, c, **kwargs):
        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        return qual, rot, width

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model


class ConvolutionalOccupancyNetworkGeometry(nn.Module):
    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()

        self.decoder_tsdf = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, inputs, p, p_tsdf, sample=True, **kwargs):
        del sample
        if isinstance(p, dict):
            _batch_size = p["p"].size(0)
        else:
            _batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        return self.decoder_tsdf(p_tsdf, c, **kwargs)

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        return self.decoder_tsdf(p_tsdf, c, **kwargs)

    def encode_inputs(self, inputs):
        if self.encoder is not None:
            return self.encoder(inputs)
        return torch.empty(inputs.size(0), 0)
