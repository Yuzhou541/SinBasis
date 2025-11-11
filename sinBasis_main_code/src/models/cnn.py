import torch.nn as nn
from .base import Classifier
from ..registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("cnn")
def build_cnn(num_classes: int = 10, **kwargs):
    return TinyCNN(num_classes=num_classes)

class TinyCNN(Classifier):
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        feats = self.net(x)
        logits = self.fc(feats.squeeze(-1).squeeze(-1))
        return logits
