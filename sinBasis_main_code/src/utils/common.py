import torch
from contextlib import nullcontext

def get_device(device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def amp_autocast(precision: str):
    if precision == "16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()
