import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    '''Synthetic dataset for quick pipeline checks (no files required).'''
    def __init__(self, num_samples=512, num_classes=10, image_size=64):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(3, self.image_size, self.image_size)
        y = torch.randint(low=0, high=self.num_classes, size=(1,)).item()
        return x, y
