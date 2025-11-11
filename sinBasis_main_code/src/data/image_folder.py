import os, glob
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageFolderDataset(Dataset):
    '''
    Simple class-per-subfolder dataset structure:
      root/
        train/
          class_a/*.jpg
          class_b/*.jpg
        val/
          class_a/*.jpg
          class_b/*.jpg
    '''
    def __init__(self, root: str, image_size: int = 224, split: str = "train"):
        super().__init__()
        self.root = os.path.join(root, split)
        self.items: List[Tuple[str, int]] = []
        classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for fp in glob.glob(os.path.join(self.root, c, "*")):
                self.items.append((fp, self.class_to_idx[c]))
        if not self.items:
            raise FileNotFoundError(f"No images found under {self.root}. Expected class subfolders with images.")
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, y = self.items[idx]
        img = Image.open(fp).convert("RGB")
        x = self.transform(img)
        return x, y
