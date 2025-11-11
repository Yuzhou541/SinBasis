# Lightweight weight-space sinusoidal reparameterization (W -> sin(W)).
# - Conv2dRaw: stores a raw parameter `W_raw` but uses `sin(W_raw)` in forward convolution.
# - LinearRaw: the same idea for Linear.
# This can help inject periodic priors for wave-like textures/spectrograms.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Classifier
from ..registry import MODEL_REGISTRY

class Conv2dRaw(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weight_raw = nn.Parameter(torch.empty(out_channels, in_channels, *k))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight_raw, nonlinearity="relu")
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight = torch.sin(self.weight_raw)  # key reparameterization
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)

class LinearRaw(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_normal_(self.weight_raw, nonlinearity="relu")

    def forward(self, x):
        w = torch.sin(self.weight_raw)
        return F.linear(x, w, self.bias)

@MODEL_REGISTRY.register("sin_cnn")
def build_sin_cnn(num_classes: int = 10, **kwargs):
    return SinCNN(num_classes=num_classes)

class SinCNN(Classifier):
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes)
        self.net = nn.Sequential(
            Conv2dRaw(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Conv2dRaw(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = LinearRaw(64, num_classes)

    def forward(self, x):
        feats = self.net(x)
        logits = self.fc(feats.squeeze(-1).squeeze(-1))
        return logits
