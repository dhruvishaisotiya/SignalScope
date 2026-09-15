"""
Bonus Module A - Faithful Explanation (heat-map + grounded text).

Since this model uses hand-crafted statistical features (not a CNN), we
cannot use Grad-CAM directly. Instead we use OCCLUSION-BASED SALIENCY,
which is model-agnostic and arguably MORE faithful than Grad-CAM because it
directly measures ground truth: "how much does the model's verdict change
if this exact region is removed?" rather than approximating with gradients.

Faithfulness guarantee: the occlusion probe re-runs visual_score() - the
EXACT fused (global + patch) score the final verdict is built from - so the
heat-map explains the real decision, not a proxy.

Method
------
1. Slide an occluding patch (mean-filled) over the image on a grid.
2. For each position, re-run the full two-branch visual scorer on the
   occluded image.
3. The drop in P(AI-generated) when a region is occluded IS that region's
   contribution to the verdict (Section 4.3 "Localisation"/"Correctness").
4. Cue-level text is produced by neutralising feature sub-groups
   (color / edge / noise / frequency / DCT) and measuring the change,
   then combined with the metadata evidence line (Module D).

Honesty (Section 4.3 "No over-claiming"): outputs are always "likely",
carry the calibrated confidence, and flag low-confidence verdicts.
"""
import cv2
import numpy as np

from features import extract_features

CUE_LABELS = {
    "color": "unnatural color/saturation statistics",
    "edge": "irregular or implausible edge sharpness",
    "noise": "missing or atypical camera sensor noise pattern",
    "freq": "anomalous high-frequency / upsampling artefacts (frequency spectrum)",
    "dct": "block-level compression artefact irregularities",
}

# feature index ranges within the 37-dim vector (see features.FEATURE_NAMES)
CUE_GROUPS = {
    "color": slice(0, 14),
    "edge": slice(14, 24),
    "noise": slice(24, 27),
    "freq": slice(27, 35),
    "dct": slice(35, 37),
}


def _visual_p(img_bgr, bundle):
    from predict import visual_score
    return visual_score(img_bgr, bundle)["p_visual"]


def occlusion_heatmap(img_bgr, bundle, grid=8, patch_frac=0.25):
    """Returns (heat [grid x grid], heat_vis normalised [0,1], base_p_visual)."""
    h, w = img_bgr.shape[:2]
    base_proba = _visual_p(img_bgr, bundle)

    mean_color = img_bgr.reshape(-1, 3).mean(axis=0)
    ph, pw = max(1, h // grid), max(1, w // grid)
    heat = np.zeros((grid, grid), dtype=np.float32)

    for gy in range(grid):
        for gx in range(grid):
            y0 = gy * ph
            y1 = (gy + 1) * ph if gy < grid - 1 else h
            x0 = gx * pw
            x1 = (gx + 1) * pw if gx < grid - 1 else w
            occluded = img_bgr.copy()
            occluded[y0:y1, x0:x1] = mean_color
            p = _visual_p(occluded, bundle)
            heat[gy, gx] = base_proba - p  # reliance of the verdict on this region

    heat_vis = heat.copy()
    if heat_vis.max() - heat_vis.min() > 1e-8:
        heat_vis = (heat_vis - heat_vis.min()) / (heat_vis.max() - heat_vis.min())
    else:
        heat_vis[:] = 0.5
    return heat, heat_vis, base_proba


def render_heatmap_overlay(img_bgr, heat_vis, out_path, upsample_size=256):
    base = cv2.resize(img_bgr, (upsample_size, upsample_size), interpolation=cv2.INTER_LINEAR)
    heat_big = cv2.resize(heat_vis, (upsample_size, upsample_size), interpolation=cv2.INTER_CUBIC)
    heat_big = np.clip(heat_big, 0, 1)
    heat_color = cv2.applyColorMap((heat_big * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.55, heat_color, 0.45, 0)
    cv2.imwrite(out_path, overlay)
    return out_path


def grounded_explanation(img_bgr, bundle, vis, meta, p_ai, threshold=0.5):
    """Cue-level explanation + metadata evidence line.
    vis: dict from predict.visual_score; meta: dict from analyze_metadata."""
    feat = extract_features(img_bgr)
    gm, gs = bundle["global_model"], bundle["global_scaler"]
    base_g = vis["p_global"]

    contributions = {}
    for name, sl in CUE_GROUPS.items():
        perturbed = gs.transform(feat.reshape(1, -1)).copy()
        perturbed[0, sl] = 0.0  # 0 std-units ~= dataset mean
        p = gm.predict_proba(perturbed)[0, 1]
        contributions[name] = abs(base_g - p)

    ranked = sorted(contributions.items(), key=lambda kv: -kv[1])
    top_cues = [CUE_LABELS[k] for k, v in ranked[:2] if v > 1e-4]

    verdict = "AI-generated" if p_ai >= threshold else "real"
    confidence = p_ai if p_ai >= threshold else 1 - p_ai

    if not top_cues:
        cue_sentence = ("No single cue dominated; the verdict rests on a diffuse "
                        "combination of low-level statistics.")
    elif len(top_cues) == 1:
        cue_sentence = f"The strongest visual signal was {top_cues[0]}."
    else:
        cue_sentence = f"The strongest visual signals were {top_cues[0]} and {top_cues[1]}."

    parts = [f"Likely {verdict} (confidence {confidence:.2f})."]

    sr = vis.get("strongest_region")
    if sr and sr != "full" and vis.get("region_scores"):
        parts.append(
            f"Composite detected: the {sr.replace('_', ' ')} of the image is "
            f"confidently AI-generated (p={vis['region_scores'][sr]:.2f}) even "
            f"though whole-image statistics are mixed - typical for collages "
            f"and side-by-side edits.")
    parts.append(cue_sentence)

    if vis.get("p_patch") is not None and vis.get("patch_weight", 0) > 0:
        if vis["p_patch"] >= 0.5:
            parts.append("Native-resolution texture crops lack a natural camera "
                         "sensor-noise signature, which supports a synthetic origin.")
        else:
            parts.append("Native-resolution texture crops carry a natural camera "
                         "sensor-noise signature, which supports a real photo.")

    parts.append(meta["summary"])
    parts.append("The highlighted regions in the heat-map show where the model's "
                 "decision would change most if that area were altered.")
    if confidence < 0.65:
        parts.append("Confidence is low, so treat this as a weak signal rather "
                     "than a firm conclusion.")

    return " ".join(parts), ranked
