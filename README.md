**Team C-433 — L. J. Institute of Engineering and Technology**
SIH-2026 Internal Hackathon · Problem Statement 2: SignalScope

| Team members      |
| ----------------- |
| Isotiya Dhruvisha |
| Gadhiya Priyanshi |
| Shah Arya         |
| Shah Dhruvi       |
| Ruparelia Sanjna  |
| Chauhan Janhvi    |


---

# SignalScope — Real vs AI-Generated Image Detector (v2)

SIH-2026 submission. Classifies an image as **real (camera)** or **AI-generated**,
with calibrated confidence, a faithful occlusion heat-map, metadata/provenance
evidence, generator-family attribution, robustness analysis, and a deployable
drag-and-drop web interface.

## What changed in v2 (the important part)

v1 had a critical flaw: **real camera photos were flagged as AI ~9 times out of 10.**

Root cause: the model was trained only on CIFAKE, whose "real" images are 32×32
CIFAR thumbnails. A high-resolution camera photo, downscaled to the analysis
size, has completely different statistics from anything in training — so it fell
on the "AI" side of the decision boundary. This is a _domain gap_, not a labeling
problem.

v2 fixes it with a **two-branch model + evidence fusion**:

| Branch                  | Input                                                                              | Trained on                                                                                                                              | Catches                                                    |
| ----------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Global**              | whole image at 32×32 (INTER_AREA), with "delivery" augmentation (re-resize + JPEG) | CIFAKE 100k (×2 augmented views = 200k rows)                                                                                            | low-res / thumbnail-scale artifacts                        |
| **Patch** (new)         | 74-dim stats over the 16 highest-variance native-resolution 32×32 crops            | 500 real photos (BSDS500) vs 256 genuine AI outputs (official sample images from Stable Diffusion, DiT, VQGAN, **SDXL and FLUX** repos) | presence/absence of real sensor noise at native resolution |
| **Metadata** (Module D) | file bytes: EXIF, PNG text chunks, XMP, C2PA markers                               | rule-based                                                                                                                              | camera provenance vs generator tags                        |

Fusion: `p_final = sigmoid((1-w)·logit(p_global) + w·logit(p_patch) + metadata_shift)`
where `w` ramps from 0 (small images) to 0.8 (≥256 px). Threshold in
`model/config.json` (default 0.5).

**Result on 16 held-out real-world photos (scikit-image sample set, never seen
in training): v1 flagged 14/16 as AI → v2 flags 5/16**, of which 3 are
non-camera content (a drawn logo, a scanned document, a scanned studio
portrait) — and real camera files also carry EXIF, which Module D uses to pull
borderline cases back toward "real".

## Metrics (held-out CIFAKE test set, 20k images)

- ROC-AUC: **0.973** (v1: 0.958)
- Macro-F1: **0.900** | Accuracy: **0.901** | FPR: 0.154
- Full numbers, ROC curve and confusion matrix: `report/metrics.json`, `report/roc_confusion.png`
- Additional honest check: patch branch evaluated on views of _source images
  excluded from training_ → AUC 0.882 (`realworld_highres_check` in metrics.json)

### Composite / collage handling (new)

Side-by-side comparisons and collages can fool whole-image statistics (crop
selection lands on seams or mixed textures). SignalScope therefore also scores
the four half-regions of the image; a half only overrides the whole-image
verdict when it is **confidently** AI (p >= 0.9), so ordinary photos are
unaffected while collages containing any clearly generated panel are flagged.
Region scores are shown in the evidence output.

## Bonus modules

| Module                     | Status         | Where                                                                                                                           |
| -------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------- | --- |
| A — explanation + heat-map | ✅             | `model/explain.py` (occlusion probes the _fused_ score, so the map explains the actual verdict)                                 |
| B — generator attribution  | ✅ (heuristic) | `model/attribution.py` — metadata tags override; else FFT checkerboard test → GAN vs diffusion family, always labeled heuristic |
| C — robustness             | ✅             | `model/robustness.py` → `report/robustness.json`, `report/degradation_chart.png`                                                |
| D — metadata / provenance  | ✅             | `model/metadata.py` — EXIF camera fields, SD/ComfyUI/Midjourney tags, XMP `trainedAlgorithmicMedia`, C2PA/JUMBF scan            |     |
| F — deployable UI          | ✅             | `app/app.py` (Flask, drag-and-drop, JSON API at `/predict`)                                                                     |
| G — adversarial analysis   | partial        | discussed in `report/model_report.md` (limitations section)                                                                     |

## Ensemble verdict (when the deep model is installed)

With `model/cnn_model.onnx` in place the verdict is an **ensemble**:
`p = sigmoid(0.55*logit(p_cnn) + 0.45*logit(p_artefact))` - the fine-tuned CNN
captures broad generator patterns while the artefact model's native-resolution
sensor-noise analysis protects real photographs. Frequency/artefact features
fused with a CNN backbone is exactly the generalisation direction §11 of the
problem statement highlights.

## Deep-detector upgrade (recommended before final judging)

The classic model generalizes reasonably, but a fine-tuned CNN generalizes to
arbitrary examiner images far better - the same class of approach commercial
detectors use. The full pipeline is included:

1. Open Google Colab (free T4 GPU), upload `training/train_deep_colab.py`
2. Follow the dataset instructions at the top of that file (CIFAKE + a modern
   AI-vs-real dataset from Kaggle + any images of your own)
3. Run it (~1-3 hours) -> it exports `cnn_model.onnx`
4. Drop that file at `model/cnn_model.onnx`

The app **auto-detects** the file and switches its primary visual scorer to
the deep model (evidence output shows `"visual_scorer": "deep-cnn"`), keeping
metadata evidence, region/collage analysis, attribution and heat-maps.
Without the file, everything runs exactly as before on the classic model.

## Run it

```bash
pip install -r requirements.txt
cd app && python app.py        # open http://localhost:5006
```

CLI:

```bash
python model/predict.py --image path/to/img.jpg --json --heatmap out.png
```

## Retrain from scratch

```bash
cd model
python build_dataset.py global --data_dir /path/to/CIFAKE --cache_dir cache --split train
python build_dataset.py global --data_dir /path/to/CIFAKE --cache_dir cache --split test
python build_dataset.py patch  --real_dir /path/to/real_photos --fake_dir /path/to/ai_images --cache_dir cache
python train.py --cache_dir cache --out signalscope_model.joblib
python evaluate.py --cache_dir cache --out_json ../report/metrics.json --out_png ../report/roc_confusion.png
python robustness.py --data_dir /path/to/CIFAKE --out_json ../report/robustness.json --out_png ../report/degradation_chart.png
python calibrate_threshold.py   # optional: re-tune decision threshold → config.json
```

## Honest limitations

- Handcrafted features + gradient boosting, not a deep network: chosen for
  CPU-only training within hackathon constraints. A fine-tuned CNN/ViT would
  outperform it.
- Patch branch trained on 256 genuine AI images spanning 2021-era latent
  diffusion through SDXL and FLUX; unseen future generators may still evade it.
- Robustness drops under heavy re-scaling (see degradation chart) — screenshots
  of screenshots reduce accuracy toward ~65%.
- Attribution (Module B) is a frequency-domain heuristic, not a trained
  classifier — labeled as such everywhere it appears.
- Output is a probabilistic likelihood, **not** a forensic certification.

## SIH submission checklist (§7)

- [x] `/app` source code, `/model` training + predict interface, `/report` model report
- [x] `requirements.txt`, setup runs a prediction in well under 10 minutes
- [x] Datasets & licences cited (above); `ORIGINALITY.md` per §8
- [x] Metrics: AUC / macro-F1 / confusion matrix / FPR at threshold in `report/`
- [x] Commits pushed within the 10-15 Sept window

Predict interface (for organizer evaluation, §4.1):
`python model/predict.py --image <path> --json` -> label + probability.
