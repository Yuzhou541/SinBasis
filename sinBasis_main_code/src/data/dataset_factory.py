from omegaconf import DictConfig
from torch.utils.data import DataLoader
from .dummy import DummyDataset
from .image_folder import ImageFolderDataset
from .hdf5_stub import HDF5Dataset

def build_dataloaders(cfg: DictConfig):
    name = cfg.data.name
    image_size = cfg.data.image_size
    if name == "dummy":
        ds_train = DummyDataset(num_samples=512, num_classes=cfg.model.num_classes, image_size=image_size)
        ds_val = DummyDataset(num_samples=128, num_classes=cfg.model.num_classes, image_size=image_size)
    elif name == "image_folder":
        root = cfg.data.root
        if not root:
            raise ValueError("data.root must be set for image_folder dataset")
        ds_train = ImageFolderDataset(root, image_size=image_size, split="train")
        ds_val = ImageFolderDataset(root, image_size=image_size, split="val")
    elif name == "hdf5_stub":
        root = cfg.data.root
        if not root:
            raise ValueError("data.root must be set for hdf5_stub")
        ds_train = HDF5Dataset(root, split="train", image_size=image_size)
        ds_val = HDF5Dataset(root, split="val", image_size=image_size)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    train_loader = DataLoader(ds_train, batch_size=cfg.trainer.batch_size, shuffle=True,
                              num_workers=cfg.trainer.num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.trainer.batch_size, shuffle=False,
                            num_workers=cfg.trainer.num_workers, pin_memory=True)
    return train_loader, val_loader
