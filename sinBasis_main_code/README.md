# AI4SCI Minimal Research Template (RTX 4090, CUDA 12.8)

A clean, **English-only** PyTorch template that trains on a dummy dataset by default (so you can verify the pipeline without any data).
Drop in your dataset later — the data layer is already abstracted.

**Why this repo?**
- Works out of the box on a single RTX 4090.
- No hard-coded CUDA minor version; just use the official *cu12* wheels.
- Config-driven (YAML) training with clean module registry.
- Includes simple baselines: MLP, CNN, ViT (timm), and a *SinBasis* reparameterized CNN (`W → sin(W)`) for wave-like signals.
- All code and docs are in English; dataset paths can be empty.

---

## 1) Environment (CUDA 12.8 driver)

> You only need an NVIDIA driver that supports CUDA 12.8 (or newer).
> PyTorch wheels ship with their own CUDA runtime, so you **do not** need to install the full toolkit.

### Option A (pip, Windows/Linux/macOS): recommended
```bash
# Create venv (Windows PowerShell)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# or Linux/macOS:
python -m venv .venv
source .venv/bin/activate

# Install PyTorch (cu12 wheels)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12

# Project deps
pip install -r requirements.txt
```

### Option B (conda / mamba)
```bash
mamba env create -f environment.yml    # or: conda env create -f environment.yml
mamba activate ai4sci-4090
```

> If you already have PyTorch installed, you can skip re-installing it.

---

## 2) Quick smoke test (no dataset needed)

```bash
# CPU/GPU check and one mini train step with DummyData
python -m src.check
```

Expected: it prints your device (CUDA if available) and runs a few quick forward/backward passes.

---

## 3) Training (config-driven)

Default config trains on a **dummy synthetic dataset** so you can verify the pipeline immediately.

```bash
# Example 1: Train a tiny CNN for 1 epoch on dummy data
python -m src.train trainer.max_epochs=1 model.name=cnn data.name=dummy

# Example 2: Use SinBasis CNN (weight-space reparam: W -> sin(W))
python -m src.train trainer.max_epochs=1 model.name=sin_cnn data.name=dummy

# Example 3: ViT (via timm)
python -m src.train trainer.max_epochs=1 model.name=vit_tiny data.name=dummy
```

To point to a real dataset later, change `data.name` and paths in `configs/default.yaml` or pass CLI overrides, e.g.
```bash
python -m src.train data.name=image_folder data.root="path/to/images" data.image_size=224
```

---

## 4) Repo layout

```
ai4sci-template-4090-cuda128/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ environment.yml
├─ configs/
│  └─ default.yaml
├─ scripts/
│  ├─ run_train.sh
│  └─ run_train.bat
└─ src/
   ├─ __init__.py
   ├─ check.py
   ├─ train.py
   ├─ infer.py
   ├─ registry.py
   ├─ utils/
   │  ├─ common.py
   │  ├─ seed.py
   │  └─ metrics.py
   ├─ data/
   │  ├─ __init__.py
   │  ├─ dataset_factory.py
   │  ├─ dummy.py
   │  ├─ image_folder.py
   │  └─ hdf5_stub.py
   └─ models/
      ├─ __init__.py
      ├─ base.py
      ├─ mlp.py
      ├─ cnn.py
      ├─ vit_tiny.py
      └─ sinbasis.py
```

- `sinbasis.py`: a lightweight **weight-space sinusoidal reparameterization** layer (`Conv2d` with raw weights; uses `sin(raw)` at forward).
  Useful for wave-like spectrograms or periodic textures.
- `image_folder.py`: simple image-folder dataset (class-per-subdir). Can be swapped out any time.
- `hdf5_stub.py`: a placeholder showing how to implement a fast HDF5-backed dataset when you add data.

---

## 5) Configuration (Hydra/OmegaConf style)

We keep it simple (no extra framework). YAML is parsed and merged with CLI overrides.

```yaml
# configs/default.yaml (excerpt)
model:
  name: "cnn"        # one of: mlp, cnn, sin_cnn, vit_tiny
  num_classes: 10
data:
  name: "dummy"      # dummy | image_folder | hdf5_stub
  root: ""           # set later for real data
  image_size: 64
trainer:
  batch_size: 32
  lr: 1e-3
  max_epochs: 10
  num_workers: 4
  device: "cuda"     # "cuda" | "cpu"
  precision: "32"    # "32" or "16" for autocast mixed precision
  log_dir: "runs"
  seed: 42
```

---

## 6) Tips for RTX 4090 + CUDA 12.8

- Ensure your **NVIDIA driver** supports CUDA 12.8+ (Device Manager → NVIDIA Control Panel → System Information).
- Prefer **AMP (mixed precision)** for speed:
  ```bash
  python -m src.train trainer.precision=16
  ```
- If you hit cudnn determinism issues, set:
  ```bash
  python -m src.train trainer.cudnn_deterministic=true
  ```

---

## 7) License

MIT
