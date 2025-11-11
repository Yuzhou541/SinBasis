# Placeholder for an HDF5-backed dataset.
# Fill in your file structure and keys when your data is ready.
import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, h5_path: str, split: str = "train", image_size: int = 64):
        self.h5_path = h5_path
        self.split = split
        self.image_size = image_size
        # NOTE: Real implementation should build an index of items for the chosen split.
        self._length = 512  # placeholder

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Example template — adapt to your file structure
        with h5py.File(self.h5_path, "r") as f:
            # x = f[f"{self.split}/images"][idx]   # shape: (H, W, 3)
            # y = f[f"{self.split}/labels"][idx]   # int
            # Placeholder synthetic tensors so the pipeline runs without real data
            x = torch.randn(3, self.image_size, self.image_size)
            y = torch.randint(low=0, high=10, size=(1,)).item()
        return x, y
