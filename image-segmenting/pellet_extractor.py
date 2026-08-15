"""
pellet_extractor.py
-------------------
Isolates the liquid pool in the bottom cone of each tube and measures its color
across multiple color spaces, for validating which channel best tracks
supernatant concentration.

DESIGN CHANGE (geometric ROI, not chroma-based):
  The previous version found the pellet BY color (saturation threshold). That
  structurally fails on faint/clear negative samples -- no color, no mask --
  which is exactly the case we must measure. The supernatant pool instead sits
  in a fixed physical location: the conical bottom of the tube. So we define the
  ROI by GEOMETRY (bottom fraction of the tube interior) and only MEASURE color
  on it. The same ROI logic applies identically to deep-red, faint, and clear
  tubes -- only the measured values differ. This is the reproducible definition
  needed for an ImageJ ground-truth comparison.

Pipeline per tube:
    1. Erode the SAM tube mask inward to shed the bright wall/glare rim
    2. Take the bottom BOTTOM_FRAC of the tube's vertical extent
    3. Intersect with the eroded mask so the ROI follows the cone taper
    4. Measure L*a*b*, HSV, red-dominance, and grayscale MGV on the ROI
       (no classification -- every tube returns a measurement)
"""

import cv2
import numpy as np


# == TUNABLE GEOMETRY =========================================================
BOTTOM_FRAC   = 0.15   # (legacy, kept for signature compatibility; the ROI now
                       #   grows by AREA via POOL_AREA_FRAC below, not height.)
POOL_AREA_FRAC = 0.09  # ROI grows upward from the tube bottom until it captures
                       #   this share of the tube's solid area. ~0.09 keeps the
                       #   ROI in the bottom liquid pool and tops out near the
                       #   meniscus instead of climbing into clear tube above it.
                       #   Calibrate against ImageJ: raise to capture more pool,
                       #   lower if it still climbs above the liquid line.
MAX_POOL_HEIGHT_FRAC = 0.18  # hard ceiling: the ROI may never extend more than
                       #   this fraction of the tube's height above its bottom,
                       #   regardless of area. Stops the ROI climbing past the
                       #   meniscus on tubes with very little liquid.
WALL_ERODE_PX = 4      # pixels to erode the tube mask inward, removing the
                       #   bright wall/specular rim that would skew color.
MIN_ROI_AREA  = 40     # below this many ROI pixels, flag as unreliable.
ROW_FILL_FRAC = 0.30   # a mask row counts as "solid tube" only if its filled
                       #   width is at least this fraction of the tube's widest
                       #   row. Thin shadow tails fall below this and are skipped
                       #   when locating the true tube bottom.
MIN_ROW_FILL_PX = 8    # absolute floor for the solid-row test (guards tiny tubes
                       #   where 30% of a narrow max width is too few pixels).
TIP_MIN_PX = 3         # a row counts as real tube (not shadow filament) if it
                       #   has at least this many pixels. Low enough that a
                       #   narrowing conical tip still registers as the bottom.
WALL_BAND_TOP_FRAC = 0.30  # wall-readout ROI: top of the body band, as a
                       #   fraction of tube height below the cap. Avoids cap
                       #   reflections.
WALL_BAND_BOT_FRAC = 0.70  # bottom of the body band, above the liquid pool.
                       #   The 0.30-0.70 band is the clean mid-wall region.


def extract_pellet(img_bgr, tube_mask, bbox,
                   bottom_frac=BOTTOM_FRAC, wall_erode_px=WALL_ERODE_PX,
                   readout="fluid"):
    """
    Define the measurement ROI for one tube.

    readout : "fluid" -> bottom liquid-pool cone (supernatant; colored = positive)
              "wall"  -> tube body side-walls above the pool (adhesion stain;
                         colored wall = NEGATIVE, clear wall = positive --
                         the OPPOSITE direction from fluid, per the paper).

    Returns a dict (always, never None -- measurement is unconditional):
        pellet_mask / valid_mask / roi_mask  full-image boolean ROI mask
        pixel_count / valid_pixel_count       ROI pixel count
        reliable     bool, False if ROI is suspiciously small
        readout      which mode produced this ROI (diagnostic)
    """
    x, y, w, h = bbox

    # 1. Erode inward to remove the bright wall/specular rim.
    mask_u8 = tube_mask.astype(np.uint8)
    if wall_erode_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * wall_erode_px + 1, 2 * wall_erode_px + 1))
        interior = cv2.erode(mask_u8, k, iterations=1).astype(bool)
    else:
        interior = tube_mask.copy()

    # Erosion can wipe a thin mask entirely; fall back to the raw mask.
    if interior.sum() < MIN_ROI_AREA:
        interior = tube_mask.copy()

    # 2. Find the tube's TRUE bottom from the UN-ERODED mask. Erosion shrinks
    #    the narrow cone tip away, so using the eroded mask would place the
    #    bottom too high (missing the pool, which sits in the tip). Measure the
    #    bottom from the original mask; measure COLOR on the eroded interior.
    raw_rows = tube_mask.sum(axis=1)
    rows_filled = interior.sum(axis=1)
    max_width = int(rows_filled.max()) if rows_filled.max() > 0 else 0
    raw_tip = np.where(raw_rows >= TIP_MIN_PX)[0]
    true_bottom = int(raw_tip.max()) if raw_tip.size else (y + h - 1)
    if max_width == 0:
        solid_top = y
    else:
        solid_thresh = max(MIN_ROW_FILL_PX, ROW_FILL_FRAC * max_width)
        solid_rows = np.where(rows_filled >= solid_thresh)[0]
        solid_top = int(solid_rows.min()) if solid_rows.size else y

    # 3. Grow the ROI UPWARD from the true bottom until it captures a target
    #    share of the solid-tube area. A fixed height-fraction fails on a
    #    conical tip (the cone narrows to a point, so the bottom slice has very
    #    few pixels and lands on the meniscus edge, not the pool body). Growing
    #    by area adapts to the cone shape: we add rows from the bottom up until
    #    the accumulated pixel count reaches POOL_AREA_FRAC of the tube area.
    # ---- WALL READOUT: measure the tube body side-walls (adhesion stain) ----
    if readout == "wall":
        # The body band runs from just below the cap to just above the pool.
        # Use the eroded interior (drops bright rim) and take a vertical band
        # of the solid-tube height, where wall adhesion is most uniform and
        # free of cap reflections and pooled liquid.
        th = max(1, true_bottom - solid_top)
        band_top = int(solid_top + WALL_BAND_TOP_FRAC * th)
        band_bot = int(solid_top + WALL_BAND_BOT_FRAC * th)
        roi = np.zeros_like(tube_mask, dtype=bool)
        roi[band_top:band_bot + 1, x:x + w] = True
        roi = roi & interior          # eroded: walls only, no bright rim
        pixel_count = int(roi.sum())
        reliable = pixel_count >= MIN_ROI_AREA
        return {
            "pellet_mask": roi, "valid_mask": roi, "roi_mask": roi,
            "pixel_count": pixel_count, "valid_pixel_count": pixel_count,
            "reliable": reliable, "readout": "wall",
        }

    # ---- FLUID READOUT: bottom liquid-pool cone (default) -------------------
    solid_mask = tube_mask.copy()
    solid_mask[:solid_top, :] = False
    solid_mask[true_bottom + 1:, :] = False
    total_solid = int(solid_mask.sum())
    target_px = max(MIN_ROI_AREA, int(POOL_AREA_FRAC * total_solid))

    row_counts = solid_mask.sum(axis=1)
    tube_height = max(1, true_bottom - solid_top)
    ceiling_row = int(true_bottom - MAX_POOL_HEIGHT_FRAC * tube_height)
    acc = 0
    cutoff_row = true_bottom
    for r in range(true_bottom, solid_top - 1, -1):   # bottom -> up
        if r < ceiling_row:        # hard height cap: never climb past meniscus
            break
        acc += int(row_counts[r])
        cutoff_row = r
        if acc >= target_px:
            break

    # 4. Build the band from cutoff_row down to the true bottom, bounded by this
    #    tube's columns. Intersect with the RAW mask (not eroded) so the narrow
    #    cone tip -- where the pool sits -- is retained. Side-wall rim removal
    #    matters less here because the pool region is mostly liquid, not wall.
    roi = np.zeros_like(tube_mask, dtype=bool)
    roi[cutoff_row:true_bottom + 1, x:x + w] = True
    roi = roi & tube_mask

    # 5. Keep only the largest connected blob. On overlapping/damaged masks the
    #    intersection can leave stray fragments (a sliver of a neighbor, or a
    #    GrabCut-clipped piece); the real liquid pool is the dominant component.
    roi_u8 = roi.astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(roi_u8)
    if n_lab > 1:
        # label 0 is background; pick the largest of the rest by area
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        roi = labels == largest

    pixel_count = int(roi.sum())
    reliable = pixel_count >= MIN_ROI_AREA

    return {
        "pellet_mask": roi,          # name kept so draw/measure helpers work
        "valid_mask": roi,
        "roi_mask": roi,
        "pixel_count": pixel_count,
        "valid_pixel_count": pixel_count,
        "reliable": reliable,
        "roi_bottom_frac": bottom_frac,
        "readout": "fluid",
    }


def measure_pellet(img_bgr, valid_mask, space="all", background_gray=None):
    """
    Measure pool color over the ROI in multiple spaces, for channel validation.

    space : "all" returns every channel; or "hsv"/"lab" for subsets.
    background_gray : optional float, mean grayscale of a tube-free background
                      ROI. If given, the grayscale-MGV baseline is
                      background-subtracted to match the paper's convention.

    Returns a dict including:
      LAB (L*, a*, b*)  -- a* is the principled redness axis
      HSV (H, S, V)
      RGB (R, G, B)
      red_dominance     -- R - (G+B)/2, a simple redness contrast
      gray_mean         -- raw mean grayscale (the paper's MGV)
      inv_mgv           -- 255 - (gray_mean - background), paper-style baseline
                           for comparison (NOT the target metric)
    """
    out = {}
    pixels_bgr = img_bgr[valid_mask]
    if pixels_bgr.size == 0:
        return out

    if space in ("lab", "all"):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)[valid_mask]
        mean = lab.mean(axis=0)
        out["LAB"] = {
            "L": float(mean[0] * 100.0 / 255.0),
            "a": float(mean[1] - 128.0),
            "b": float(mean[2] - 128.0),
        }

    if space in ("hsv", "all"):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[valid_mask]
        mean = hsv.mean(axis=0)
        out["HSV"] = {"H": float(mean[0]), "S": float(mean[1]), "V": float(mean[2])}

    mean_bgr = pixels_bgr.mean(axis=0)
    R, G, B = float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])
    out["RGB"] = {"R": R, "G": G, "B": B}

    # Simple redness contrast: how much red exceeds the avg of green+blue.
    out["red_dominance"] = R - (G + B) / 2.0

    # Grayscale MGV baseline (the paper's metric) for cross-comparison only.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[valid_mask]
    gray_mean = float(gray.mean())
    out["gray_mean"] = gray_mean
    if background_gray is not None:
        out["inv_mgv"] = 255.0 - (gray_mean - background_gray)
    else:
        out["inv_mgv"] = 255.0 - gray_mean

    return out


def draw_pellets(img_rgb, tubes, pellet_results):
    """Outline each ROI on a copy of the image."""
    vis = img_rgb.copy()
    overlay = vis.copy()
    for i, (t, pr) in enumerate(zip(tubes, pellet_results)):
        if pr is None:
            continue
        overlay[pr["pellet_mask"]] = (255, 60, 60)
        ys, xs = np.where(pr["pellet_mask"])
        if len(xs):
            cv2.putText(vis, f"T{i+1}", (xs.min(), ys.min() - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 2)
    return cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)