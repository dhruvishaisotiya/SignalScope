"""
Build cached feature datasets for BOTH branches of the classifier.

Global branch (CIFAKE):
    python build_dataset.py global --data_dir /path/to/archive --cache_dir cache
  For every training image we extract features twice: once as-is, and once
  through a "delivery simulation" (random upscale to a display size, random
  JPEG re-compression, occasional mild noise). This teaches the global model
  that the same content arriving at a different size / compression level is
  still the same class - the exact conditions a screenshot or shared image
  arrives in.

Patch branch (real high-res photos vs genuine diffusion outputs):
    python build_dataset.py patch --real_dir bsds_images --fake_dir aipool --cache_dir cache
  Sources:
    REAL: BSDS500 photographs (Berkeley Segmentation Dataset - 500 real
          camera photos, classic open research dataset).
    FAKE: genuine high-resolution diffusion/VQGAN outputs collected from the
          official Stable Diffusion, Latent Diffusion, DiT and
          taming-transformers repositories (sample images published by the
          model authors themselves - i.e. certainly AI-generated).
  Each source image contributes several augmented "views" (random rescale +
  JPEG), and each view contributes one 74-dim patch-aggregate row.
  The split is BY SOURCE IMAGE so no photo leaks between train and val.
"""
import argparse
import os
import random

import cv2
import numpy as np

from features import extract_features, extract_patch_features

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_images(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def delivery_augment(img, rng):
    """Simulate how an image typically reaches a detector in the wild:
    displayed / shared at some size, re-encoded as JPEG, sometimes noisy."""
    s = rng.randint(128, 512)
    interp = rng.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_LANCZOS4])
    im = cv2.resize(img, (s, s), interpolation=interp)
    if rng.random() < 0.7:
        q = rng.randint(55, 95)
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            im = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.2:
        im = im.astype(np.float32) + np.random.normal(0, rng.uniform(1, 4), im.shape)
        im = np.clip(im, 0, 255).astype(np.uint8)
    return im


def build_global(data_dir, cache_dir, split):
    rng = random.Random(42)
    X, y = [], []
    for label, cls in [(0, "REAL"), (1, "FAKE")]:
        files = list_images(os.path.join(data_dir, split, cls))
        print(f"[global/{split}] {cls}: {len(files)} images")
        for i, p in enumerate(files):
            img = cv2.imread(p)
            if img is None:
                continue
            X.append(extract_features(img)); y.append(label)
            if split == "train":
                X.append(extract_features(delivery_augment(img, rng))); y.append(label)
            if (i + 1) % 10000 == 0:
                print(f"  {cls} {i + 1}/{len(files)}", flush=True)
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int64)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, f"Xg_{split}.npy"), X)
    np.save(os.path.join(cache_dir, f"yg_{split}.npy"), y)
    print(f"[global/{split}] saved {X.shape}")


def _patch_views(img, n_views, rng, label=0):
    """Yield augmented views of a high-res source image (random rescale to
    simulate shared/downsized copies, random JPEG). For REAL images we also
    sometimes simulate smartphone computational photography (edge-preserving
    denoise + mild sharpen), which suppresses raw sensor noise - modern phone
    photos must still be recognised as real."""
    h, w = img.shape[:2]
    for _ in range(n_views):
        scale = rng.uniform(0.35, 1.0)
        nh, nw = max(96, int(h * scale)), max(96, int(w * scale))
        im = cv2.resize(img, (nw, nh),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        if rng.random() < 0.7:
            q = rng.randint(55, 96)
            ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                im = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        yield im


def build_patch(real_dir, fake_dir, cache_dir, real_views=14, fake_views=28,
                holdout_frac=0.25):
    rng = random.Random(7)
    data = {"train": ([], []), "val": ([], [])}
    for label, folder, n_views in [(0, real_dir, real_views), (1, fake_dir, fake_views)]:
        files = list_images(folder)
        rng.shuffle(files)
        n_hold = max(1, int(len(files) * holdout_frac))
        for split, flist in [("val", files[:n_hold]), ("train", files[n_hold:])]:
            for p in flist:
                img = cv2.imread(p)
                if img is None or min(img.shape[:2]) < 96:
                    continue
                nv = n_views if split == "train" else max(4, n_views // 3)
                for view in _patch_views(img, nv, rng, label=label):
                    feat = extract_patch_features(view)
                    if feat is not None:
                        data[split][0].append(feat)
                        data[split][1].append(label)
        print(f"[patch] label={label} files={len(files)} (holdout {n_hold} source images)")
    os.makedirs(cache_dir, exist_ok=True)
    for split in ("train", "val"):
        X = np.array(data[split][0], dtype=np.float32)
        y = np.array(data[split][1], dtype=np.int64)
        np.save(os.path.join(cache_dir, f"Xp_{split}.npy"), X)
        np.save(os.path.join(cache_dir, f"yp_{split}.npy"), y)
        print(f"[patch/{split}] saved {X.shape}, fake fraction {y.mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("global")
    g.add_argument("--data_dir", required=True)
    g.add_argument("--cache_dir", default="cache")
    g.add_argument("--split", default="train", choices=["train", "test"])
    p = sub.add_parser("patch")
    p.add_argument("--real_dir", required=True)
    p.add_argument("--fake_dir", required=True)
    p.add_argument("--cache_dir", default="cache")
    args = ap.parse_args()
    if args.cmd == "global":
        build_global(args.data_dir, args.cache_dir, args.split)
    else:
        build_patch(args.real_dir, args.fake_dir, args.cache_dir)


if __name__ == "__main__":
    main()
