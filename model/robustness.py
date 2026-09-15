"""
Bonus Module C - Robustness to Degradation.

Re-evaluates the fused visual classifier on a sample of held-out images
after applying real-world degradations (JPEG re-compression at several
qualities, downscale/upscale "resize", and a screenshot simulation), and
reports how accuracy shifts.

    python robustness.py --data_dir /path/to/archive --model signalscope_model.joblib \
        --out_json ../report/robustness.json --out_png ../report/degradation_chart.png
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from joblib import load


def degrade_jpeg(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def degrade_resize(img, scale=0.5):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_screenshot(img):
    """Simulate a screenshot: slight blur + resize round-trip + mediocre
    JPEG + minor brightness shift."""
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, (3, 3), 0.6)
    resized = cv2.resize(blurred, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized.astype(np.float32) * 1.05 + 3, 0, 255).astype(np.uint8)
    return degrade_jpeg(resized, quality=60)


DEGRADATIONS = [
    ("clean", lambda im: im),
    ("jpeg_q80", lambda im: degrade_jpeg(im, 80)),
    ("jpeg_q60", lambda im: degrade_jpeg(im, 60)),
    ("jpeg_q40", lambda im: degrade_jpeg(im, 40)),
    ("resize_0.5x", lambda im: degrade_resize(im, 0.5)),
    ("screenshot_sim", degrade_screenshot),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="archive dir with test/REAL, test/FAKE")
    ap.add_argument("--model", default="signalscope_model.joblib")
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_png", default=None)
    args = ap.parse_args()

    from predict import visual_score
    bundle = load(args.model)

    files, labels = [], []
    for lbl, cls in [(0, "REAL"), (1, "FAKE")]:
        fs = sorted(glob.glob(os.path.join(args.data_dir, "test", cls, "*")))[: args.n_per_class]
        files += fs; labels += [lbl] * len(fs)
    labels = np.array(labels)
    imgs = [cv2.imread(p) for p in files]

    results = {}
    for name, fn in DEGRADATIONS:
        proba = np.array([visual_score(fn(im), bundle)["p_visual"] for im in imgs])
        acc = float(((proba >= args.threshold) == labels).mean())
        results[name] = {"accuracy": acc}
        print(f"{name:16s} acc={acc:.4f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
    if args.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(results); accs = [results[n]["accuracy"] for n in names]
        plt.figure(figsize=(8, 4))
        plt.bar(names, accs, color="#4a7ebb")
        plt.ylim(0, 1); plt.ylabel("Accuracy"); plt.xticks(rotation=20)
        plt.title(f"Accuracy under degradation (held-out sample, n={len(files)})")
        plt.tight_layout(); plt.savefig(args.out_png, dpi=130)
        print("Saved chart to", args.out_png)


if __name__ == "__main__":
    main()
