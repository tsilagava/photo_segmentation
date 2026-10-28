"""U-Net with an ImageNet-pretrained ResNet-34 encoder (same architecture as the training notebooks)."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_bn_relu(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(c_in + c_skip, c_out), conv_bn_relu(c_out, c_out))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # 1/2,  64 ch
        self.pool = r.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = r.layer1, r.layer2, r.layer3, r.layer4  # 1/4 .. 1/32
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)
        e1 = self.enc1(self.pool(s0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d = self.dec4(e4, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, s0)
        return self.head(self.dec0(d))


def load_segmenter(path, device="cpu"):
    """Loads a checkpoint saved by train.py / the notebooks (or exported by scripts/export_weights.py).

    Returns (model in eval mode, training input size, checkpoint dict without the weights).
    """
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    class_names = ck.get("class_names", ["background", "pet"])
    net = UNetResNet34(len(class_names), pretrained=False)
    net.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    meta = {k: v for k, v in ck.items() if k != "model"}
    return net.to(device).eval(), int(ck.get("config", {}).get("img_size", 256)), meta
