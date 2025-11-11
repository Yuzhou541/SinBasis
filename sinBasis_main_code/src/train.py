import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .registry import MODEL_REGISTRY
from .data.dataset_factory import build_dataloaders
from .utils.seed import set_seed
from .utils.common import get_device, amp_autocast

def build_model(cfg):
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    image_size = cfg.data.image_size
    builder = MODEL_REGISTRY.get(name)
    model = builder(num_classes=num_classes, image_size=image_size)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, precision):
    model.train()
    total_loss = 0.0
    total_seen = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), torch.as_tensor(y, device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(precision):
            logits = model(x)
            loss = criterion(logits, y)
        if precision == "16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += x.size(0)
    return total_loss / total_seen, correct / total_seen

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), torch.as_tensor(y, device=device, dtype=torch.long)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += x.size(0)
    return total_loss / total_seen, correct / total_seen

def main():
    cfg = OmegaConf.load("configs/default.yaml")
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)

    set_seed(cfg.trainer.seed, deterministic_cudnn=cfg.trainer.get("cudnn_deterministic", False))
    device = get_device(cfg.trainer.device)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.trainer.precision == "16"))
    writer = SummaryWriter(log_dir=cfg.trainer.log_dir)

    best_val_acc = 0.0
    for epoch in range(cfg.trainer.max_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, cfg.trainer.precision)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(f"[Epoch {epoch+1}/{cfg.trainer.max_epochs}]"
              f" train_loss={train_loss:.4f} acc={train_acc:.3f} |"
              f" val_loss={val_loss:.4f} acc={val_acc:.3f} ({dt:.1f}s)")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)},
                       f"checkpoints/best.pt")

    writer.close()
    print(f"Best val acc: {best_val_acc:.3f}")

if __name__ == "__main__":
    main()
