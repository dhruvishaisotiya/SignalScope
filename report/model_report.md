# SignalScope v2 — Model Report

## 1. Problem
Binary classification: real (camera) vs AI-generated image, with calibrated
confidence, explanation, and provenance evidence (SIH-2026 problem statement).

## 2. Why v1 failed on real photos
v1 trained a RandomForest on 37 handcrafted features extracted at 64×64 from
CIFAKE only. CIFAKE "real" images are 32×32 CIFAR thumbnails; real camera
photos (4000+ px) downscaled to 64×64 produce feature vectors far outside the
training distribution and were classified "AI" ~88% of the time
(14/16 on a held-out real-photo set).

## 3. v2 architecture
Two visual branches + metadata evidence, fused in logit space.

**Global branch.** 37 features (color statistics, Laplacian sharpness, Sobel
edge histogram, high-pass noise residual, FFT radial energy profile, DCT AC
statistics) at 32×32 — CIFAKE's *native* size — using INTER_AREA when
shrinking. Training uses *delivery augmentation*: each of the 100k CIFAKE train
images is also re-sampled to a random 128–512 px size and JPEG-recompressed
(q 55–95), teaching the branch what "the same image after web delivery" looks
like. 200k rows total. Classifier: HistGradientBoosting + isotonic calibration
(fit/calibration/validation split 70/15/15).

**Patch branch (the fix).** For images ≥96 px min-dim: 16 deterministic
highest-variance 32×32 crops at native resolution → mean+std of the 37
features = 74-dim vector. Trained on 500 BSDS500 camera photos vs 256 genuine
generator outputs collected from the official Stable Diffusion, latent
diffusion, DiT, VQGAN, SDXL (Stability generative-models) and FLUX
(black-forest-labs) repositories (grids split into tiles, visually
verified). 25% of *source images* held out entirely: held-out source AUC 0.882.
This branch keys on real sensor noise, which survives at native resolution and
which diffusion decoders do not reproduce.

**Fusion.** w ramps 0→0.8 between 96 and 256 px min-dim:
`p = sigmoid((1-w)·logit(p_global) + w·logit(p_patch) + metadata_shift)`.
Metadata shift (Module D): +4.0 log-odds for explicit generator tags,
−2.2 for ≥2 strong camera EXIF fields, −1.0 for one.

## 4. Results
| Test | v1 | v2 |
|---|---|---|
| CIFAKE test ROC-AUC | 0.958 | **0.973** |
| CIFAKE test macro-F1 | — | **0.900** |
| Real photos flagged as AI (16 held-out) | 14 | **5** (3 are scans/drawn graphics) |
| Genuine SD/DiT outputs flagged as AI | — | 89/102 |
| CIFAKE fakes (native res) flagged | — | 92/100 |
| CIFAKE fakes upscaled to 512 + JPEG | — | **100/100** |

Confusion matrix and ROC: `roc_confusion.png`. Degradation: `degradation_chart.png`
(accuracy holds to JPEG q60, drops for 0.5× re-scaling / screenshot simulation).

## 5. Explanation faithfulness (Module A)
The occlusion heat-map perturbs 8×8 regions and re-scores with the *same fused
visual score* used for the verdict, so hot regions are causally tied to the
decision, not a separate saliency proxy. Text explanations quote the measured
cue groups (frequency profile, noise residual, edge statistics), the patch
branch's sensor-noise finding, and metadata evidence; low-margin verdicts are
hedged explicitly.

## 6. Known gaps / adversarial notes (Module G, partial)
- An attacker can add synthetic sensor-like noise to defeat the patch branch;
  frequency-profile features are partially resistant but not robust to targeted
  optimization.
- Metadata is trivially strippable/forgeable; it is used only as a soft
  log-odds shift, never as a sole verdict.
- Training fakes span 2021-era diffusion through SDXL/FLUX; unreleased future generators may evade.
- Screenshot-of-screenshot pipelines degrade accuracy to ~65%.

## 7. Data sources
- CIFAKE (user-provided archive.zip) — global branch train/test.
- BSDS500 (Berkeley) — real high-res photos, patch branch.
- Official sample images from CompVis/stable-diffusion,
  CompVis/taming-transformers, facebookresearch/DiT, Stability-AI/generative-models (SDXL),
  black-forest-labs/flux — genuine AI outputs,
  patch branch. Grids split to tiles; ambiguous tiles discarded.
- scikit-image sample photos — untouched real-world validation only.
