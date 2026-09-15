"""
Train both branches of the SignalScope classifier and save one model bundle.

    python train.py --cache_dir cache --out signalscope_model.joblib

Global branch: HistGradientBoosting on 37-dim whole-image features
  (CIFAKE train split, with delivery augmentation), probabilities
  calibrated isotonically on a held-aside calibration split.
Patch branch:  HistGradientBoosting on 74-dim native-crop aggregates
  (BSDS500 real photos vs genuine diffusion outputs), validated on
  source images never seen in training.

The bundle stores {global_model, global_scaler, patch_model, patch_scaler}
and is what predict.py / the web app load.
"""
import argparse
import os

import numpy as np
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_branch(X, y, name, max_iter=400):
    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest)

    scaler = StandardScaler().fit(X_fit)
    base = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1, max_depth=None,
        l2_regularization=0.5, random_state=42, class_weight="balanced")
    base.fit(scaler.transform(X_fit), y_fit)

    clf = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    clf.fit(scaler.transform(X_cal), y_cal)

    proba = clf.predict_proba(scaler.transform(X_val))[:, 1]
    auc = roc_auc_score(y_val, proba)
    f1 = f1_score(y_val, (proba >= 0.5).astype(int), average="macro")
    print(f"[{name}] internal validation: AUC={auc:.4f}  macro-F1={f1:.4f}  (n={len(y_val)})")
    return clf, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out", default="signalscope_model.joblib")
    args = ap.parse_args()

    Xg = np.load(os.path.join(args.cache_dir, "Xg_train.npy"))
    yg = np.load(os.path.join(args.cache_dir, "yg_train.npy"))
    print(f"[global] training rows: {Xg.shape}")
    gm, gs = train_branch(Xg, yg, "global")

    Xp = np.load(os.path.join(args.cache_dir, "Xp_train.npy"))
    yp = np.load(os.path.join(args.cache_dir, "yp_train.npy"))
    print(f"[patch] training rows: {Xp.shape}")
    pm, ps = train_branch(Xp, yp, "patch", max_iter=300)

    # honest patch validation: SOURCE IMAGES never seen in training
    Xpv = np.load(os.path.join(args.cache_dir, "Xp_val.npy"))
    ypv = np.load(os.path.join(args.cache_dir, "yp_val.npy"))
    proba = pm.predict_proba(ps.transform(Xpv))[:, 1]
    print(f"[patch] held-out SOURCE-IMAGE validation: "
          f"AUC={roc_auc_score(ypv, proba):.4f}  "
          f"acc@0.5={((proba >= 0.5) == ypv).mean():.4f}  (n={len(ypv)})")

    dump({"global_model": gm, "global_scaler": gs,
          "patch_model": pm, "patch_scaler": ps}, args.out, compress=3)
    print("Saved bundle to", args.out)


if __name__ == "__main__":
    main()
