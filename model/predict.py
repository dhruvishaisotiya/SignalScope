"""
REQUIRED PREDICT INTERFACE (Section 4.1 of the problem statement).

CLI:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --json
    python predict.py --image path/to/image.jpg --heatmap out.png

Python:
    from predict import predict, predict_proba
    predict("img.jpg")        -> "AI-generated" | "real"
    predict_proba("img.jpg")  -> dict with p_ai, confidence, verdict,
                                 explanation, evidence breakdown, ...

HOW THE VERDICT IS FORMED (evidence fusion in log-odds space)
--------------------------------------------------------------
1. Global branch  - whole image at 32x32 (trained on CIFAKE + delivery
                    augmentation). Always used.
2. Patch branch   - top-K native-resolution crops (trained on real high-res
                    photos vs genuine diffusion outputs). Weight ramps from
                    0 for thumbnail inputs to 0.8 for images >= ~256 px,
                    where native sensor-noise evidence is most reliable.
3. Metadata       - EXIF camera fields shift toward "real"; AI-tool tags /
                    C2PA trainedAlgorithmicMedia shift strongly toward
                    "AI-generated" (Bonus Module D). Capped and reported.

p_final = sigmoid( (1-w)*logit(p_global) + w*logit(p_patch) + meta_shift )
"""
import argparse
import json
import os

import cv2
import numpy as np
from joblib import load

from features import extract_features, extract_patch_features, patch_branch_weight
from metadata import analyze_metadata
from attribution import attribute_generator

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "signalscope_model.joblib")
CONFIG_PATH = os.path.join(HERE, "config.json")

_cache = {}
EPS = 1e-5


def _get_bundle(model_path=DEFAULT_MODEL):
    if model_path not in _cache:
        _cache[model_path] = load(model_path)
    return _cache[model_path]


def _default_threshold() -> float:
    """Decision threshold, re-tunable via calibrate_threshold.py -> config.json."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return float(json.load(f).get("threshold", 0.5))
        except Exception:
            pass
    return 0.5


def _logit(p):
    p = float(np.clip(p, EPS, 1 - EPS))
    return float(np.log(p / (1 - p)))


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def visual_score(img_bgr, bundle) -> dict:
    """Fused VISUAL P(AI) for an in-memory image (no metadata - metadata
    belongs to the file, not the pixels). Used by predict_proba and by the
    occlusion explainer, so the heat-map explains exactly the score the
    verdict uses."""
    gf = extract_features(img_bgr).reshape(1, -1)
    p_g = float(bundle["global_model"].predict_proba(
        bundle["global_scaler"].transform(gf))[0, 1])

    w = patch_branch_weight(img_bgr)
    p_p = None
    if w > 0:
        pf = extract_patch_features(img_bgr)
        if pf is not None:
            p_p = float(bundle["patch_model"].predict_proba(
                bundle["patch_scaler"].transform(pf.reshape(1, -1)))[0, 1])
        else:
            w = 0.0
    L = (1 - w) * _logit(p_g) + (w * _logit(p_p) if p_p is not None else 0.0)
    return {"p_visual": _sigmoid(L), "p_global": p_g, "p_patch": p_p, "patch_weight": w}


REGION_MIN_DIM = 200  # only sub-analyze images big enough for meaningful halves


from cnn_infer import cnn_available, cnn_p_ai


def visual_score_regions(img_bgr, bundle) -> dict:
    """Composite-aware visual score. Collages / side-by-side comparisons can
    fool a whole-image analysis (crop selection lands on seams or mixed
    textures), so we also score the left/right and top/bottom halves and take
    the strongest AI evidence: if ANY region is confidently AI-generated, the
    image contains AI-generated content."""
    use_cnn = cnn_available()
    if use_cnn:
        # Ensemble: deep CNN (broad visual patterns) fused with the classic
        # two-branch artefact model (sensor-noise / frequency cues that
        # generalise differently). Each covers the other's blind spots -
        # in particular the classic patch branch protects real high-res
        # photos, while the CNN generalises across generator styles.
        full = dict(visual_score(img_bgr, bundle))
        p_classic = full["p_visual"]
        p_cnn = cnn_p_ai(img_bgr)
        full["p_classic"] = p_classic
        full["p_cnn"] = p_cnn
        full["p_visual"] = _sigmoid(0.55 * _logit(p_cnn) + 0.45 * _logit(p_classic))
        full["scorer"] = "ensemble (deep-cnn + artefact model)"
    else:
        full = visual_score(img_bgr, bundle)
        full = dict(full); full["scorer"] = "classic"

    def _region_p(part):
        if not use_cnn:
            return visual_score(part, bundle)["p_visual"]
        return _sigmoid(0.55 * _logit(cnn_p_ai(part))
                        + 0.45 * _logit(visual_score(part, bundle)["p_visual"]))

    regions = {"full": full["p_visual"]}
    h, w = img_bgr.shape[:2]
    if min(h, w) >= REGION_MIN_DIM:
        halves = {
            "left_half": img_bgr[:, : w // 2],
            "right_half": img_bgr[:, w // 2:],
            "top_half": img_bgr[: h // 2, :],
            "bottom_half": img_bgr[h // 2:, :],
        }
        for name, part in halves.items():
            regions[name] = _region_p(part)
    strongest = max(regions, key=regions.get)
    out = dict(full)
    # A sub-region only overrides the whole-image verdict when it is
    # CONFIDENTLY AI (>=0.9). Half of a normal photo can be mildly atypical
    # (sky, bokeh, texture); demanding strong evidence keeps the false-positive
    # rate on ordinary photos unchanged while still catching collages where
    # one panel is clearly generated.
    if strongest != "full" and regions[strongest] >= 0.90 and full["p_visual"] < regions[strongest]:
        out["p_visual"] = regions[strongest]
    out["region_scores"] = {k: round(v, 4) for k, v in regions.items()}
    out["strongest_region"] = strongest if out["p_visual"] == regions[strongest] else "full"
    return out


def predict_proba(image_path: str, model_path: str = DEFAULT_MODEL,
                  threshold: float = None, with_explanation: bool = True) -> dict:
    if threshold is None:
        threshold = _default_threshold()
    bundle = _get_bundle(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    vis = visual_score_regions(img, bundle)
    meta = analyze_metadata(image_path)
    L = _logit(vis["p_visual"]) + meta["log_odds_shift"]
    p_ai = _sigmoid(L)

    label = "AI-generated" if p_ai >= threshold else "real"
    confidence = p_ai if label == "AI-generated" else 1 - p_ai

    result = {
        "image": image_path,
        "label": label,
        "p_ai": round(p_ai, 4),
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "verdict": f"Likely {label}",
        "evidence": {
            "visual_p_ai": round(vis["p_visual"], 4),
            "global_branch_p_ai": round(vis["p_global"], 4),
            "patch_branch_p_ai": (round(vis["p_patch"], 4)
                                  if vis["p_patch"] is not None else None),
            "patch_branch_weight": round(vis["patch_weight"], 2),
            "visual_scorer": vis.get("scorer", "classic"),
            "region_scores": vis.get("region_scores"),
            "strongest_region": vis.get("strongest_region"),
            "metadata_summary": meta["summary"],
            "metadata_log_odds_shift": meta["log_odds_shift"],
            "camera_exif": meta["camera_evidence"][:6],
            "ai_metadata": meta["ai_evidence"][:6],
        },
        "note": "Probabilistic likelihood estimate, not a certified determination.",
    }

    # Bonus Module B: attribution, only meaningful for AI verdicts
    if label == "AI-generated":
        result["generator_attribution"] = attribute_generator(img, meta)

    if with_explanation:
        from explain import occlusion_heatmap, grounded_explanation
        heat, heat_vis, _ = occlusion_heatmap(img, bundle, grid=8)
        explanation, cue_ranking = grounded_explanation(img, bundle, vis, meta, p_ai, threshold)
        result["explanation"] = explanation
        result["cue_ranking"] = [{"cue": k, "influence": round(float(v), 4)}
                                 for k, v in cue_ranking]
        result["_heat_vis"] = heat_vis  # stripped before JSON output

    return result


def predict(image_path: str, model_path: str = DEFAULT_MODEL) -> str:
    """Returns the label string 'real' or 'AI-generated'."""
    return predict_proba(image_path, model_path, with_explanation=False)["label"]


def main():
    ap = argparse.ArgumentParser(description="SignalScope: real vs AI-generated image classifier")
    ap.add_argument("--image", required=True, help="path to an image file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold in config.json (default 0.5)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--heatmap", default=None, help="optional path to save the explanation heat-map PNG")
    args = ap.parse_args()

    result = predict_proba(args.image, args.model, args.threshold)
    heat_vis = result.pop("_heat_vis", None)

    if args.heatmap and heat_vis is not None:
        from explain import render_heatmap_overlay
        img = cv2.imread(args.image)
        render_heatmap_overlay(img, heat_vis, args.heatmap)
        result["heatmap_saved_to"] = args.heatmap

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['verdict']}  (P(AI)={result['p_ai']:.3f}, confidence={result['confidence']:.3f})")
        print(f"  {result['explanation']}")
        ev = result["evidence"]
        print(f"  visual={ev['visual_p_ai']:.3f} "
              f"(global={ev['global_branch_p_ai']:.3f}, "
              f"patch={ev['patch_branch_p_ai']}, w={ev['patch_branch_weight']})")
        print(f"  metadata: {ev['metadata_summary']}")
        if "generator_attribution" in result:
            at = result["generator_attribution"]
            print(f"  attribution: {at['family']} - {at['basis']}")
        if args.heatmap:
            print(f"  heat-map saved to {args.heatmap}")


if __name__ == "__main__":
    main()
