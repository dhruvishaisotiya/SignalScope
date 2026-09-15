"""
Evaluate the trained model on the held-out TEST split (never used in
training), plus a real-world domain check on source images the model has
never seen.

    python evaluate.py --cache_dir cache --model signalscope_model.joblib \
        --out_json ../report/metrics.json --out_png ../report/roc_confusion.png

Reports: ROC-AUC, macro-F1, confusion matrix, accuracy & FPR at the
operating threshold - both for the CIFAKE held-out test set (core metric,
global branch) and for the patch branch's held-out source images
(real BSDS photos / genuine diffusion outputs never used in training).
"""
import argparse
import json
import os

import numpy as np
from joblib import load
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--model", default="signalscope_model.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_png", default=None)
    args = ap.parse_args()

    bundle = load(args.model)

    # ---- Core: CIFAKE held-out test split (global branch) ----
    Xt = np.load(os.path.join(args.cache_dir, "Xg_test.npy"))
    yt = np.load(os.path.join(args.cache_dir, "yg_test.npy"))
    proba = bundle["global_model"].predict_proba(
        bundle["global_scaler"].transform(Xt))[:, 1]
    pred = (proba >= args.threshold).astype(int)

    cm = confusion_matrix(yt, pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "core_test_set": "CIFAKE held-out test split (20k images, never trained on)",
        "n_test": int(len(yt)),
        "roc_auc": float(roc_auc_score(yt, proba)),
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "accuracy_at_threshold": float(accuracy_score(yt, pred)),
        "threshold": args.threshold,
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "confusion_matrix": cm.tolist(),
    }

    # ---- Real-world domain check: patch-branch held-out SOURCE images ----
    Xpv = np.load(os.path.join(args.cache_dir, "Xp_val.npy"))
    ypv = np.load(os.path.join(args.cache_dir, "yp_val.npy"))
    pp = bundle["patch_model"].predict_proba(
        bundle["patch_scaler"].transform(Xpv))[:, 1]
    metrics["realworld_highres_check"] = {
        "what": ("views of real photos (BSDS500) and genuine diffusion outputs "
                 "whose SOURCE IMAGES were excluded from training"),
        "n": int(len(ypv)),
        "roc_auc": float(roc_auc_score(ypv, pp)),
        "accuracy_at_threshold": float(((pp >= args.threshold) == ypv).mean()),
    }

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)

    if args.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(yt, proba)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
        axes[0].set_title("ROC - CIFAKE held-out test"); axes[0].legend()
        im = axes[1].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            axes[1].text(j, i, str(v), ha="center", va="center",
                         color="white" if v > cm.max() / 2 else "black")
        axes[1].set_xticks([0, 1], ["real", "AI-gen"]); axes[1].set_yticks([0, 1], ["real", "AI-gen"])
        axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
        axes[1].set_title(f"Confusion @ threshold {args.threshold}")
        fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
        print("Saved plot to", args.out_png)


if __name__ == "__main__":
    main()
