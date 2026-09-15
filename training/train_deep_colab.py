"""
SignalScope Deep Detector - Google Colab GPU training script
=============================================================
Trains an EfficientNet-B0 real-vs-AI classifier at 224px and exports it to
ONNX. The SignalScope app automatically uses `model/cnn_model.onnx` as its
primary visual scorer when the file is present (falling back to the classic
two-branch model otherwise).

HOW TO RUN (Google Colab, free tier is enough):
  1. Open https://colab.research.google.com -> New notebook
  2. Runtime -> Change runtime type -> T4 GPU
  3. Upload this file, or paste its contents into a cell prefixed with %%writefile train_deep_colab.py
  4. Get the datasets (see DATASETS below), then run:  !python train_deep_colab.py
  5. After ~1-3 hours it saves cnn_model.onnx -> download it and place it at
     signalscope/model/cnn_model.onnx on your machine. Done.

DATASETS (the more diverse, the better the examiner-proofing):
  Required:
    - CIFAKE: kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
  Strongly recommended (adds modern generators + high-res real photos):
    - kaggle datasets download -d alessandrasala79/ai-vs-human-generated-dataset
    - Any folder of your own real phone photos (put in data/extra_real/)
    - Any folder of confirmed AI images you collected (put in data/extra_fake/)
  Arrange as:
    data/
      real/   <- all real images (any size, any format)
      fake/   <- all AI images
In Colab:
  !pip -q install kaggle
  # upload your kaggle.json (Kaggle -> Account -> Create API token)
  !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  !kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data_raw --unzip
  # then move/merge folders into data/real and data/fake
"""
import os
import random
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image, ImageFilter
import io

DATA_DIR = os.environ.get("DATA_DIR", "data")
EPOCHS = int(os.environ.get("EPOCHS", 4))
BATCH = int(os.environ.get("BATCH", 64))
LR = float(os.environ.get("LR", 3e-4))
IMG = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)


class JpegCompress:
    """Random JPEG recompression - teaches the net what web delivery does."""
    def __call__(self, img):
        if random.random() < 0.5:
            q = random.randint(45, 95)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class RandomRescale:
    """Random down/up scaling - screenshots, thumbnails, re-shares."""
    def __call__(self, img):
        if random.random() < 0.5:
            w, h = img.size
            s = random.uniform(0.3, 1.0)
            img = img.resize((max(32, int(w * s)), max(32, int(h * s))),
                             Image.BILINEAR)
        return img


train_tf = T.Compose([
    RandomRescale(),
    JpegCompress(),
    T.RandomResizedCrop(IMG, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.Lambda(lambda im: im.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.0)))
             if random.random() < 0.2 else im),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = T.Compose([
    T.Resize(256), T.CenterCrop(IMG), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FolderPairs(Dataset):
    def __init__(self, files, labels, tf):
        self.files, self.labels, self.tf = files, labels, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG, IMG))
        return self.tf(img), self.labels[i]


def collect():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    real, fake = [], []
    for e in exts:
        real += glob.glob(os.path.join(DATA_DIR, "real", "**", e), recursive=True)
        fake += glob.glob(os.path.join(DATA_DIR, "fake", "**", e), recursive=True)
    print(f"real={len(real)}  fake={len(fake)}")
    files = real + fake
    labels = [0] * len(real) + [1] * len(fake)
    idx = list(range(len(files)))
    random.Random(42).shuffle(idx)
    files = [files[i] for i in idx]
    labels = [labels[i] for i in idx]
    n_val = max(1000, len(files) // 20)
    return (files[n_val:], labels[n_val:]), (files[:n_val], labels[:n_val])


def main():
    (trf, trl), (vaf, val) = collect()
    tr = DataLoader(FolderPairs(trf, trl, train_tf), batch_size=BATCH,
                    shuffle=True, num_workers=2, pin_memory=True)
    va = DataLoader(FolderPairs(vaf, val, val_tf), batch_size=BATCH,
                    num_workers=2, pin_memory=True)

    net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
    net = net.to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(tr))
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")

    best_auc = 0.0
    for ep in range(EPOCHS):
        net.train()
        for bi, (x, y) in enumerate(tr):
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
                out = net(x).squeeze(1)
                loss = lossf(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if bi % 100 == 0:
                print(f"ep{ep} {bi}/{len(tr)} loss={loss.item():.4f}")
        # validation AUC
        net.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in va:
                p = torch.sigmoid(net(x.to(DEVICE)).squeeze(1)).cpu()
                ps += p.tolist()
                ys += y.tolist()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(ys, ps)
        print(f"== epoch {ep}: val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(net.state_dict(), "best.pth")

    net.load_state_dict(torch.load("best.pth"))
    net.eval().cpu()
    dummy = torch.randn(1, 3, IMG, IMG)
    torch.onnx.export(net, dummy, "cnn_model.onnx",
                      input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input": {0: "batch"}},
                      opset_version=17)
    print(f"Exported cnn_model.onnx (best val AUC {best_auc:.4f}).")
    print("Copy it to signalscope/model/cnn_model.onnx - the app picks it up automatically.")


if __name__ == "__main__":
    main()
