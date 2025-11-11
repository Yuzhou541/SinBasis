import torch
from .utils.common import get_device
from .train import build_model
from omegaconf import OmegaConf

def main():
    cfg = OmegaConf.load("configs/default.yaml")
    device = get_device(cfg.trainer.device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    model = build_model(cfg).to(device)
    x = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size, device=device)
    y = model(x)
    print("Forward OK | output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
