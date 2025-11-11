import timm
from ..registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("vit_tiny")
def build_vit_tiny(num_classes: int = 10, image_size: int = 224, **kwargs):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    return model
