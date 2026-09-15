"""
Bonus Module B - Generator-family attribution (heuristic).

Given an image the visual model believes is AI-generated, estimate the
likely generator FAMILY. Honesty first: our training data contains
diffusion-family fakes only (CIFAKE = Stable Diffusion; patch branch =
SD / LDM / DiT / VQGAN), so this module is a transparent frequency-domain
HEURISTIC, not a trained multi-class model, and it is always labelled as
such in the output.

Signal used - GAN upsampling fingerprint:
  Transposed-convolution / PixelShuffle upsampling in classic GANs leaves
  periodic "checkerboard" artefacts that appear as distinct PEAKS at
  fractions of the sampling frequency (f/2, f/4) in the azimuthally-averaged
  FFT spectrum (Zhang et al., "Detecting and Simulating Artifacts in GAN
  Fake Images", WIFS 2019). Diffusion decoders produce a smoother spectral
  decay without those localised peaks. We measure peak prominence at the
  expected frequencies and report:
      strong peaks   -> "GAN-family (checkerboard spectrum peaks)"
      no clear peaks -> "diffusion-family (smooth spectral decay)"

If metadata already names the tool (Module D), that overrides the
heuristic - the tag IS the attribution.
"""
import cv2
import numpy as np


def _radial_profile(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray * np.hanning(gray.shape[0])[:, None]
                                    * np.hanning(gray.shape[1])[None, :]))
    mag = np.log(np.abs(f) + 1)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.int32)
    prof = np.bincount(r.ravel(), mag.ravel()) / (np.bincount(r.ravel()) + 1e-9)
    return prof[: min(cy, cx)]


def attribute_generator(img_bgr: np.ndarray, metadata: dict | None = None) -> dict:
    """Returns {family, basis, confidence_note}."""
    # Metadata override: an explicit tool tag is the best attribution we have.
    if metadata and metadata.get("ai_evidence"):
        ev = " / ".join(metadata["ai_evidence"][:2])
        return {
            "family": "named by metadata",
            "basis": f"generator tool named in file metadata ({ev})",
            "confidence_note": "metadata-based; high confidence unless the file was deliberately re-tagged",
        }

    side = 256
    img = cv2.resize(img_bgr, (side, side), interpolation=cv2.INTER_AREA
                     if min(img_bgr.shape[:2]) > side else cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # high-pass residual isolates the upsampling fingerprint
    resid = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    prof = _radial_profile(resid)
    n = len(prof)
    smooth = np.convolve(prof, np.ones(9) / 9, mode="same")
    prominence = prof - smooth

    # look for localised peaks near f/2 and f/4 of the analysis grid
    peak_score = 0.0
    for frac in (0.5, 0.25):
        idx = int(n * frac)
        lo, hi = max(1, idx - 4), min(n - 1, idx + 4)
        peak_score = max(peak_score, float(prominence[lo:hi].max()))
    noise_floor = float(np.abs(prominence[n // 8: n // 2]).mean() + 1e-9)
    ratio = peak_score / noise_floor

    if ratio > 6.0:
        family = "GAN-family (heuristic)"
        basis = "distinct periodic peaks in the high-frequency spectrum, characteristic of transposed-convolution upsampling"
    else:
        family = "diffusion-family (heuristic)"
        basis = "smooth high-frequency spectral decay without checkerboard peaks, typical of diffusion/VAE decoders"

    return {
        "family": family,
        "basis": basis,
        "confidence_note": ("heuristic estimate from frequency analysis only; "
                            "the detector was trained on diffusion-family fakes, "
                            "so treat attribution as indicative, not definitive"),
    }
