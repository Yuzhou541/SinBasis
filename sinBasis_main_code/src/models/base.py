import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError
