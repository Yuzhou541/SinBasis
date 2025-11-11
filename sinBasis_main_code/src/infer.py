import torch, json
from omegaconf import OmegaConf
from .registry import MODEL_REGISTRY
from .utils.common import get_device

def load_model(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = OmegaConf.create(state["cfg"])
    builder = MODEL_REGISTRY.get(cfg.model.name)
    model = builder(num_classes=cfg.model.num_classes, image_size=cfg.data.image_size)
    model.load_state_dict(state["model"], strict=True)
    return model, cfg

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoints/best.pt")
    ap.add_argument("--image", type=str, required=False, help="Optional image path for single prediction")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt)
    device = get_device(cfg.trainer.device)
    model.to(device).eval()

    if args.image:
        from PIL import Image
        import torchvision.transforms as T
        img = Image.open(args.image).convert("RGB")
        tfm = T.Compose([T.Resize((cfg.data.image_size, cfg.data.image_size)), T.ToTensor()])
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1).item()
        print(json.dumps({"pred_class": int(pred)}, indent=2))
    else:
        print("Model loaded. Provide --image to run a single prediction.")

if __name__ == "__main__":
    main()
