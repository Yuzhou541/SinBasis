import torch.nn as nn
from .base import Classifier
from ..registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("mlp")
def build_mlp(num_classes: int = 10, image_size: int = 64, **kwargs):
    flat = 3 * image_size * image_size
    return TinyMLP(flat, num_classes)

class TinyMLP(Classifier):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__(num_classes=num_classes)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
