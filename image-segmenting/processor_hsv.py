"""
processor_hsv.py
----------------
Measures tip ROI color using the HSV color space.

HSV is used for BOTH artifact masking (bubbles, highlights, shadows)
and color measurement. Best suited when color differences are large
and intuitive hue-based reasoning is preferred.
"""

import cv2
import numpy as np


# ── ARTIFACT MASK THRESHOLDS (tunable) ────────────────────────────────────────
BUBBLE_SAT_MAX   = 30    # pixels below this saturation are bubble/highlight
HIGHLIGHT_VAL_MIN = 230  # pixels above this brightness are specular highlights
SHADOW_VAL_MAX   = 30    # pixels below this brightness are shadows


def _build_artifact_mask(tip_hsv):
    """
    Return a boolean mask (True = artifact, exclude this pixel).
    Catches bubbles, specular highlights, and shadows.
    """
    S = tip_hsv[:, :, 1]
    V = tip_hsv[:, :, 2]
    return (S < BUBBLE_SAT_MAX) | (V > HIGHLIGHT_VAL_MIN) | (V < SHADOW_VAL_MAX)


def measure(img_bgr, tip_coords):
    """
    Measure the mean HSV color of the valid (non-artifact) pixels
    inside the tip ROI.

    Parameters
    ----------
    img_bgr    : np.ndarray   full BGR image
    tip_coords : tuple        (x1, y1, x2, y2) tip ROI coordinates

    Returns
    -------
    dict with keys:
        H, S, V             mean HSV values (OpenCV ranges: H 0-180, S/V 0-255)
        valid_pixel_count   number of pixels used in measurement
        method              "HSV"
    Returns None if no valid pixels remain after masking.
    """
    x1, y1, x2, y2 = tip_coords

    tip_bgr = img_bgr[y1:y2, x1:x2]
    if tip_bgr.size == 0:
        return None

    tip_hsv = cv2.cvtColor(tip_bgr, cv2.COLOR_BGR2HSV)
    artifact_mask = _build_artifact_mask(tip_hsv)
    valid_mask = ~artifact_mask

    valid_pixels = tip_hsv[valid_mask]
    if len(valid_pixels) == 0:
        return None

    mean_hsv = valid_pixels.mean(axis=0)

    return {
        "method": "HSV",
        "H": mean_hsv[0],
        "S": mean_hsv[1],
        "V": mean_hsv[2],
        "valid_pixel_count": int(valid_mask.sum()),
    }


def format_result(result):
    """Human-readable string for terminal output."""
    return (f"[HSV] H={result['H']:.2f}  S={result['S']:.2f}  "
            f"V={result['V']:.2f}  | valid px={result['valid_pixel_count']}")