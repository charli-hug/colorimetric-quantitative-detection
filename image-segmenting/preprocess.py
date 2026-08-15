"""
preprocess.py
-------------
Optional pre-SAM background/shadow suppression using GrabCut.

WHY: On gray/low-contrast backgrounds, cast shadows are contiguous with the
tubes and SAM bleeds the shadow into the bottom of each tube mask -- right where
the liquid-pool ROI lives, which corrupts color measurements. Masking the
background (and its shadows) to black BEFORE SAM inserts a hard tube/black edge
so SAM cuts cleanly at the true tube boundary.

GrabCut is used (rather than a fixed brightness/saturation threshold) because
the imaging lighting will vary across sessions; GrabCut re-estimates the
foreground/background color models per image and adapts automatically.

This is OPT-IN. Clean, high-contrast backgrounds (e.g. white) don't need it and
should skip it to avoid the iterative cost and any edge artifacts.

Auto initialization: assume a border margin of the image is background and the
interior is probable-foreground, then let GrabCut refine. No manual ROI needed.
"""

import cv2
import numpy as np


def suppress_background(img_bgr, border_frac=0.04, iters=5, fill=(0, 0, 0)):
    """
    Return a copy of img_bgr with background+shadow set to `fill`.

    Parameters
    ----------
    img_bgr     : np.ndarray  BGR image
    border_frac : float       fraction of width/height treated as definite
                              background at the image edges (GrabCut init).
                              Keep small; only the outer rim is assumed bg.
    iters       : int         GrabCut iterations (5 is plenty).
    fill        : tuple       BGR color to paint the background. Black (0,0,0)
                              gives SAM a hard edge to cut on.

    Returns
    -------
    masked_img : np.ndarray  image with background suppressed
    fg_mask    : np.ndarray  boolean foreground mask (True = kept)
    """
    H, W = img_bgr.shape[:2]

    # GrabCut mask init: everything probable-background, interior probable-fg.
    gc_mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)

    bx = max(1, int(W * border_frac))
    by = max(1, int(H * border_frac))
    # Outer rim = definite background.
    gc_mask[:by, :] = cv2.GC_BGD
    gc_mask[-by:, :] = cv2.GC_BGD
    gc_mask[:, :bx] = cv2.GC_BGD
    gc_mask[:, -bx:] = cv2.GC_BGD
    # Interior = probable foreground.
    gc_mask[by:-by, bx:-bx] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model,
                iters, cv2.GC_INIT_WITH_MASK)

    # Definite + probable foreground -> keep.
    fg_mask = np.isin(gc_mask, [cv2.GC_FGD, cv2.GC_PR_FGD])

    # Clean up speckle: keep only sizeable foreground components, close holes.
    fg_u8 = fg_mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, k, iterations=1)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, k, iterations=2)
    fg_mask = fg_u8.astype(bool)

    masked = img_bgr.copy()
    masked[~fg_mask] = fill
    return masked, fg_mask