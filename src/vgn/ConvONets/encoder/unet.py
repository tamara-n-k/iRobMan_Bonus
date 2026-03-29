import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
    return nn.Sequential(
        nn.Upsample(mode="bilinear", scale_factor=2),
        conv1x1(in_channels, out_channels),
    )


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1,
    )


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.pooling = pooling
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode="concat", up_mode="transpose"):
        super().__init__()
        self.merge_mode = merge_mode
        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)

        if self.merge_mode == "concat":
            self.conv1 = conv3x3(2 * out_channels, out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=3,
        depth=5,
        start_filts=64,
        up_mode="transpose",
        merge_mode="concat",
        **kwargs,
    ):
        super().__init__()

        if up_mode not in ("transpose", "upsample"):
            raise ValueError(f"{up_mode!r} is not a valid upsampling mode")
        if merge_mode not in ("concat", "add"):
            raise ValueError(f"{merge_mode!r} is not a valid merge mode")
        if up_mode == "upsample" and merge_mode == "add":
            raise ValueError("up_mode='upsample' is incompatible with merge_mode='add'")

        self.down_convs = []
        self.up_convs = []
        outs = None

        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filts * (2**i)
            pooling = i < depth - 1
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for _ in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.conv_final = conv1x1(outs, num_classes)
        self.reset_params()

    @staticmethod
    def weight_init(module):
        if isinstance(module, nn.Conv2d):
            init.xavier_normal_(module.weight)
            init.constant_(module.bias, 0)

    def reset_params(self):
        for module in self.modules():
            self.weight_init(module)

    def forward(self, x):
        encoder_outs = []

        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for module, before_pool in zip(self.up_convs, reversed(encoder_outs[:-1])):
            x = module(before_pool, x)

        return self.conv_final(x)
