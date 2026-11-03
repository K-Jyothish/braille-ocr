"""Detector networks. Both take RGB in [0, 1], shape (B, 3, H, W), and return logits (B, 1, H, W)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class PlainUNet(nn.Module):
    """The original project's U-Net (3 poolings, 64-512 channels, grayscale input).

    Kept layer-for-layer so the baseline is the same architecture, retrained on
    the new data with crops at native resolution instead of 256x256 page resizes.
    Input sides must be multiples of 8.
    """

    multiple = 8

    def __init__(self):
        super().__init__()

        def block(in_c, out_c):  # original layout, conv bias included, for state-dict compatibility
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )

        self.enc1, self.pool1 = block(1, 64), nn.MaxPool2d(2)
        self.enc2, self.pool2 = block(64, 128), nn.MaxPool2d(2)
        self.enc3, self.pool3 = block(128, 256), nn.MaxPool2d(2)
        self.bottleneck = block(256, 512)
        self.up3, self.dec3 = nn.ConvTranspose2d(512, 256, 2, stride=2), block(512, 256)
        self.up2, self.dec2 = nn.ConvTranspose2d(256, 128, 2, stride=2), block(256, 128)
        self.up1, self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2), block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


class _Up(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.conv = _block(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class ResNetUNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet encoder and a full-resolution output.

    Encoder features at strides 2-32 feed a bilinear-upsampling decoder; a
    shallow full-resolution branch provides the stride-1 skip so dot centres
    can be localised to the pixel. Input sides must be multiples of 32.
    """

    multiple = 32

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, prior: float = 0.01):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        r = getattr(torchvision.models, backbone)(weights=weights)
        chans = [64, 64, 128, 256, 512] if backbone in ("resnet18", "resnet34") else [64, 256, 512, 1024, 2048]
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)            # /2
        self.pool = r.maxpool                                        # /4
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.full = _block(3, 32)                                    # /1
        self.up4 = _Up(chans[4], chans[3], 256)                      # -> /16
        self.up3 = _Up(256, chans[2], 128)                           # -> /8
        self.up2 = _Up(128, chans[1], 64)                            # -> /4
        self.up1 = _Up(64, chans[0], 64)                             # -> /2
        self.up0 = _Up(64, 32, 32)                                   # -> /1
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.head.bias, float(torch.logit(torch.tensor(prior))))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = (rgb - self.mean) / self.std
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        d = self.up4(s5, s4)
        d = self.up3(d, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, self.full(x))
        return self.head(d)


MODELS = {
    "resnet_heatmap": ("heatmap", lambda **kw: ResNetUNet(**kw)),
    "plain_unet": ("mask", lambda **kw: PlainUNet()),
}


def build_model(name: str, **kwargs) -> tuple[nn.Module, str]:
    """Returns (model, target kind) for a name in :data:`MODELS`."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    kind, ctor = MODELS[name]
    return ctor(**kwargs), kind
