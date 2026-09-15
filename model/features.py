"""
SignalScope - Feature Extraction Module (v2, two-branch)
=========================================================
Classic computer-vision / frequency-domain feature extractor used for the
real-vs-AI-generated image classifier.

WHY v2 EXISTS - the domain-gap bug and its fix
-----------------------------------------------
v1 analysed every image by resizing it to 64x64 with bicubic interpolation.
The training data (CIFAKE) is 32x32, so real CIFAR thumbnails reached the
model as 32->64 *upscales* (interpolation-smooth), while a user's real
phone/camera photo reached it as a 4000->64 bicubic *downscale* (aliased,
totally different statistics). Result: ~90% of real camera photos were
flagged "AI-generated". Two changes fix this:

1. GLOBAL BRANCH at 32x32 with the right resampling.
   We analyse at CIFAKE's native 32x32 and use INTER_AREA (box averaging)
   when shrinking - the same family of operation used to make CIFAR
   thumbnails from real photos in the first place - so a downscaled phone
   photo and a CIFAR "real" now live in the same statistical space.

2. PATCH BRANCH at native resolution (new).
   High-resolution images carry the strongest evidence at the *pixel* level:
   real camera photos contain sensor noise in every 32x32 native crop, while
   diffusion/GAN decoder output does not. We take the top-K highest-detail
   native 32x32 crops (deterministic, content-adaptive), extract the same
   37 features from each, and aggregate (mean + std across crops -> 74 dims).
   A second classifier trained on real high-res photos (BSDS500) vs genuine
   diffusion outputs (Stable Diffusion / LDM / DiT / VQGAN samples) scores
   this branch. It is only used when the image is big enough to have native
   crops (min side >= PATCH_MIN_DIM), and its weight grows with resolution.

The 37 per-view features (unchanged from v1, rationale):
  - Color statistics        -> generator outputs have subtly different
                               channel mean/std/saturation distributions.
  - Edge / Laplacian energy -> real photos have natural blur falloff and
                               noise; synthetic images are often "too clean".
  - Sobel gradient histogram-> shape of the edge-strength distribution.
  - High-frequency noise residual -> camera sensor noise signature that
                               generative decoders fail to reproduce.
  - FFT radial energy       -> upsampling / decoder artefacts show up as
                               anomalous energy at specific frequencies.
  - 8x8 block DCT AC energy -> JPEG-style block statistics differ between
                               camera pipelines and generator outputs.

None of these depend on image *content*, only on how the pixels were
produced - which is what gives the approach a fighting chance on unseen
generators.
"""

import cv2
import numpy as np

FEATURE_NAMES = (
    ["mean_B", "std_B", "p10_B", "p90_B",
     "mean_G", "std_G", "p10_G", "p90_G",
     "mean_R", "std_R", "p10_R", "p90_R",
     "sat_mean", "sat_std",
     "laplacian_var", "laplacian_abs_mean"]
    + [f"sobel_hist_{i}" for i in range(8)]
    + ["noise_std", "noise_mean", "noise_abs_mean"]
    + [f"fft_radial_{i}" for i in range(8)]
    + ["dct_ac_mean", "dct_ac_std"]
)

ANALYSIS_DIM = 32     # global branch: CIFAKE's native resolution
PATCH_SIZE = 32       # patch branch: native-resolution crop size
PATCH_GRID = 6        # candidate grid for crop selection
PATCH_TOPK = 16       # number of crops aggregated
PATCH_MIN_DIM = 96    # patch branch only used when min(h, w) >= this


def smart_resize(img_bgr: np.ndarray, dim: int = ANALYSIS_DIM) -> np.ndarray:
    """Resize with INTER_AREA when shrinking (box average, like thumbnail
    creation) and INTER_CUBIC when enlarging. Using the right resampling
    direction is what keeps a downscaled phone photo in the same statistical
    space as the CIFAR-style training reals."""
    h, w = img_bgr.shape[:2]
    interp = cv2.INTER_AREA if min(h, w) > dim else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (dim, dim), interpolation=interp)


def _stats_features(img_f32: np.ndarray) -> np.ndarray:
    """37 features from a DIM x DIM float32 BGR view."""
    dim = img_f32.shape[0]
    gray = cv2.cvtColor(img_f32.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    feats = []

    # --- Color statistics per channel (B, G, R) ---
    for c in range(3):
        ch = img_f32[:, :, c]
        feats += [ch.mean(), ch.std(), np.percentile(ch, 10), np.percentile(ch, 90)]

    # --- Saturation stats (HSV) ---
    hsv = cv2.cvtColor(img_f32.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    feats += [hsv[:, :, 1].mean(), hsv[:, :, 1].std()]

    # --- Edge / sharpness (Laplacian) ---
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    feats += [lap.var(), np.mean(np.abs(lap))]

    # --- Sobel gradient magnitude histogram ---
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    hist, _ = np.histogram(mag, bins=8, range=(0, 255))
    feats += (hist / (hist.sum() + 1e-6)).tolist()

    # --- High-frequency noise residual ---
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    resid = gray - blur
    feats += [resid.std(), resid.mean(), np.mean(np.abs(resid))]

    # --- FFT radial energy spectrum (8 concentric bins) ---
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag_spec = np.log(np.abs(fshift) + 1)
    h, w = mag_spec.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.int32)
    nbins = 8
    bin_edges = np.linspace(0, r.max(), nbins + 1)
    for i in range(nbins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        feats.append(mag_spec[mask].mean() if mask.sum() > 0 else 0.0)

    # --- 8x8 block DCT AC-coefficient energy ---
    ac_energies = []
    for by in range(0, dim, 8):
        for bx in range(0, dim, 8):
            block = gray[by:by + 8, bx:bx + 8]
            dct = cv2.dct(block)
            ac = np.abs(dct).sum() - np.abs(dct[0, 0])
            ac_energies.append(ac)
    ac_energies = np.array(ac_energies)
    feats += [ac_energies.mean(), ac_energies.std()]

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Global branch
# ---------------------------------------------------------------------------

def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """37-dim global feature vector (whole image resized to 32x32)."""
    return _stats_features(smart_resize(img_bgr).astype(np.float32))


def extract_features_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return extract_features(img)


# ---------------------------------------------------------------------------
# Patch branch (native-resolution evidence for high-res images)
# ---------------------------------------------------------------------------

def select_patch_boxes(img_bgr: np.ndarray, grid: int = PATCH_GRID,
                       topk: int = PATCH_TOPK, size: int = PATCH_SIZE):
    """Deterministic, content-adaptive crop selection: split the image into
    a grid, rank cells by local detail (gray std), take a size x size native
    crop at the centre of the top-k cells. Deterministic so that the
    occlusion heat-map is meaningful and predictions are reproducible."""
    h, w = img_bgr.shape[:2]
    if min(h, w) < size:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ch, cw = h // grid, w // grid
    cells = []
    for gy in range(grid):
        for gx in range(grid):
            cell = gray[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
            cells.append((float(cell.std()), gy, gx))
    cells.sort(reverse=True)
    boxes = []
    for _, gy, gx in cells[:topk]:
        cy = gy * ch + ch // 2
        cx = gx * cw + cw // 2
        y0 = int(np.clip(cy - size // 2, 0, h - size))
        x0 = int(np.clip(cx - size // 2, 0, w - size))
        boxes.append((y0, x0, size))
    return boxes


def extract_patch_features(img_bgr: np.ndarray) -> np.ndarray | None:
    """74-dim aggregate (mean + std of the 37 features over the top-K
    native crops). Returns None when the image is too small for native
    crops (then only the global branch is used)."""
    if min(img_bgr.shape[:2]) < PATCH_MIN_DIM:
        return None
    boxes = select_patch_boxes(img_bgr)
    if len(boxes) < 4:
        return None
    per_crop = []
    for (y0, x0, s) in boxes:
        crop = img_bgr[y0:y0 + s, x0:x0 + s].astype(np.float32)
        per_crop.append(_stats_features(crop))
    per_crop = np.stack(per_crop)
    return np.concatenate([per_crop.mean(axis=0), per_crop.std(axis=0)]).astype(np.float32)


def patch_branch_weight(img_bgr: np.ndarray) -> float:
    """How much the fused visual verdict trusts the patch branch.
    0.0 for thumbnail-sized inputs (CIFAKE-style: global branch only,
    so held-out benchmark behaviour is exactly the global model),
    ramping to 0.8 for images >= ~256 px, where native-resolution
    sensor-noise evidence is the most reliable signal."""
    d = min(img_bgr.shape[:2])
    if d < PATCH_MIN_DIM:
        return 0.0
    return 0.8 * float(np.clip((d - 64) / (256 - 64), 0.0, 1.0))
