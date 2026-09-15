"""
(Optional) Re-tune the real/AI decision threshold on YOUR OWN images.

The shipped default threshold is 0.5 and works out of the box. If you find
the balance wrong for your particular image sources, run this with a folder
of known-real photos and a folder of known-AI images; it picks the
threshold that maximises balanced accuracy on your data and writes it to
model/config.json, which predict.py and the web app pick up automatically.

    python calibrate_threshold.py --real_dir my_real_photos --fake_dir my_ai_images
"""
import argparse
import glob
import json
import os

import numpy as np

from predict import predict_proba, DEFAULT_MODEL

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
HERE = os.path.dirname(os.path.abspath(__file__))


def _load_images(folder):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(folder, ext))
        files += glob.glob(os.path.join(folder, ext.upper()))
    return sorted(set(files))


def _score_folder(folder):
    scores = []
    for p in _load_images(folder):
        try:
            scores.append(predict_proba(p, with_explanation=False)["p_ai"])
        except Exception as e:
            print(f"  skip {p}: {e}")
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    args = ap.parse_args()

    s_real = _score_folder(args.real_dir)
    s_fake = _score_folder(args.fake_dir)
    print(f"real: n={len(s_real)} mean p_ai={s_real.mean():.3f}")
    print(f"fake: n={len(s_fake)} mean p_ai={s_fake.mean():.3f}")

    best_t, best_bal = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        bal = 0.5 * ((s_real < t).mean() + (s_fake >= t).mean())
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    print(f"best threshold={best_t:.3f} balanced-accuracy={best_bal:.3f}")

    cfg_path = os.path.join(HERE, "config.json")
    with open(cfg_path, "w") as f:
        json.dump({"threshold": round(best_t, 3)}, f, indent=2)
    print("wrote", cfg_path)


if __name__ == "__main__":
    main()
