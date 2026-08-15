"""
processor_lab.py
----------------
Measures tip ROI color using the CIELAB color space.

Artifact masking is still done in HSV (most intuitive for brightness-based
exclusion). Color measurement is done in LAB, which is perceptually uniform —
numerical differences correspond to visible differences, making it better
for quantitative colorimetric assays and building calibration curves.
"""

import cv2
import numpy as np


# ── ARTIFACT MASK THRESHOLDS (shared logic with HSV processor) ─────────────────
BUBBLE_SAT_MAX    = 30    # HSV saturation below this = bubble/highlight
HIGHLIGHT_VAL_MIN = 230   # HSV value above this = specular highlight
SHADOW_VAL_MAX    = 30    # HSV value below this = shadow


def _build_artifact_mask(tip_hsv):
    """
    Return a boolean mask (True = artifact, exclude this pixel).
    Uses HSV for masking even though measurement is in LAB —
    HSV brightness/saturation is more intuitive for artifact detection.
    """
    S = tip_hsv[:, :, 1]
    V = tip_hsv[:, :, 2]
    return (S < BUBBLE_SAT_MAX) | (V > HIGHLIGHT_VAL_MIN) | (V < SHADOW_VAL_MAX)


def measure(img_bgr, tip_coords):
    """
    Measure the mean CIELAB color of the valid (non-artifact) pixels
    inside the tip ROI.

    Parameters
    ----------
    img_bgr    : np.ndarray   full BGR image
    tip_coords : tuple        (x1, y1, x2, y2) tip ROI coordinates

    Returns
    -------
    dict with keys:
        L, a, b             true CIELAB values (L* 0-100, a*/b* -128 to 127)
        valid_pixel_count   number of pixels used in measurement
        method              "CIELAB"
    Returns None if no valid pixels remain after masking.
    """
    x1, y1, x2, y2 = tip_coords

    tip_bgr = img_bgr[y1:y2, x1:x2]
    if tip_bgr.size == 0:
        return None

    # Build artifact mask in HSV
    tip_hsv = cv2.cvtColor(tip_bgr, cv2.COLOR_BGR2HSV)
    artifact_mask = _build_artifact_mask(tip_hsv)
    valid_mask = ~artifact_mask

    # Measure in LAB
    tip_lab = cv2.cvtColor(tip_bgr, cv2.COLOR_BGR2Lab)
    valid_pixels = tip_lab[valid_mask]

    if len(valid_pixels) == 0:
        return None

    # OpenCV compresses LAB into [0, 255] — rescale to true ranges
    mean_lab_raw = valid_pixels.mean(axis=0)
    L = mean_lab_raw[0] * (100.0 / 255.0)   # L*: 0–100
    a = mean_lab_raw[1] - 128.0              # a*: -128–127
    b = mean_lab_raw[2] - 128.0              # b*: -128–127

    return {
        "method": "CIELAB",
        "L": L,
        "a": a,
        "b": b,
        "valid_pixel_count": int(valid_mask.sum()),
    }


def delta_e(result_a, result_b):
    """
    Compute CIE76 ΔE between two LAB measurement results.
    ΔE < 1 is imperceptible; ΔE > 2-3 is visually noticeable.
    """
    return np.sqrt(
        (result_a["L"] - result_b["L"]) ** 2 +
        (result_a["a"] - result_b["a"]) ** 2 +
        (result_a["b"] - result_b["b"]) ** 2
    )


def format_result(result):
    """Human-readable string for terminal output."""
    return (f"[LAB] L*={result['L']:.2f}  a*={result['a']:.2f}  "
            f"b*={result['b']:.2f}  | valid px={result['valid_pixel_count']}")