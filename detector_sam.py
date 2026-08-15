"""
detector_sam.py
---------------
Tube detection using Segment Anything (MobileSAM / SAM ViT-B).

SAM produces a clean mask per object regardless of background, lighting,
transparency, or whether tubes are touching. We then filter the returned masks
down to tube-shaped ones, AND deduplicate overlapping/nested masks so a single
tube can't be counted multiple times (cap mask + body mask + full-tube mask).

KEY CHANGES vs. the previous version:
  * SamAutomaticMaskGenerator retuned for a few large, well-separated objects:
      - points_per_side  64 -> 32   (dense seeding was fragmenting tubes)
      - crop_n_layers      1 -> 0    (cropping just produced more duplicates)
      - pred_iou_thresh        -> 0.80   (high values drop clear/faint tubes)
      - stability_score_thresh -> 0.84
  * detect_tubes now runs a CONTAINMENT/OVERLAP dedup pass after shape
    filtering. This is what fixes "12 masks for 6 tubes": nested cap/body masks
    are collapsed to the single best mask per tube.
  * Each tube now carries its SAM quality score + area for dedup scoring and
    debugging.

SETUP (run once in your venv):
    # Standard SAM:
    pip install git+https://github.com/facebookresearch/segment-anything.git
    # Download vit_b weights: sam_vit_b_01ec64.pth
    #
    # OR MobileSAM:
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git
    # Download mobile_sam.pt
"""

import cv2
import numpy as np


def load_sam(model_type="vit_b", checkpoint="sam_vit_b_01ec64.pth", device=None):
    """
    Load a SAM automatic mask generator.

    model_type : "mobile_sam" -> lightweight (uses vit_t registry, mobile_sam pkg)
                 "vit_b"/"vit_l"/"vit_h" -> standard SAM (segment_anything pkg)
    checkpoint : path to the .pt / .pth weight file
    device     : "cpu", "mps", "cuda", or None to auto-pick
    """
    import torch

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"[SAM] Using device: {device}")

    if model_type == "mobile_sam":
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        print(f"[SAM] Loaded MobileSAM (vit_t) from {checkpoint}")
    else:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        print(f"[SAM] Loaded segment-anything '{model_type}' from {checkpoint}")

    sam.to(device=device)
    sam.eval()

    # Tuned so faint/transparent tubes still register. Clear tubes produce
    # lower-confidence masks; thresholds set too high silently drop them -- and
    # those are the negative/low-signal samples the assay most needs. We keep
    # quality moderate and rely on the dedup + shape filters to remove the extra
    # masks that looser thresholds let through.
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,             # was 64 -- dense seeding fragmented tubes
        pred_iou_thresh=0.84,           # balanced: clear tubes survive, fewer
                                        #   fragment masks than 0.80
        stability_score_thresh=0.86,    # was 0.92 -- 0.92 dropped clear tubes
        min_mask_region_area=1500,
        crop_n_layers=0,                # was 1 -- cropping produced duplicates
        crop_n_points_downscale_factor=2,
        box_nms_thresh=0.5,             # tighter built-in NMS between masks
    )
    return mask_generator


def _mask_bbox_area(seg):
    x, y, w, h = cv2.boundingRect(seg)
    return x, y, w, h


def _overlap_ratio(mask_a, mask_b):
    """
    Returns intersection / area_of_smaller_mask.
    High value => one mask is largely contained in the other (e.g. a cap mask
    sitting inside a full-tube mask). This catches nesting that plain IoU misses.
    """
    inter = np.logical_and(mask_a, mask_b).sum()
    smaller = min(mask_a.sum(), mask_b.sum())
    return inter / smaller if smaller > 0 else 0.0


def _dedup_masks(candidates, overlap_thresh=0.5):
    """
    Collapse overlapping/nested candidate masks to one mask per object.

    candidates : list of dicts with keys: mask, bbox, area, score
    Strategy   : sort best-first (area * score), then greedily accept a mask
                 only if it doesn't substantially overlap an already-accepted
                 one. The first (best) mask of each tube wins; its cap/body
                 fragments are discarded.
    """
    ordered = sorted(candidates, key=lambda c: c["area"] * c["score"], reverse=True)
    kept = []
    for cand in ordered:
        overlaps_existing = False
        for k in kept:
            if _overlap_ratio(cand["mask"], k["mask"]) > overlap_thresh:
                overlaps_existing = True
                break
        if not overlaps_existing:
            kept.append(cand)
    return kept


def detect_tubes(img, mask_generator,
                 area_min=4000, aspect_ratio_min=1.6, aspect_ratio_max=4.5,
                 solidity_min=0.55, extent_min=0.30,
                 max_area_fraction=0.6,
                 overlap_thresh=0.5,
                 single_tube=False,
                 debug=False):
    """
    Run SAM on a BGR image and filter + dedup masks down to tube objects.

    New parameters
    --------------
    overlap_thresh : float  if two surviving masks overlap by more than this
                            (intersection / smaller-mask area), keep only the
                            better one. Lower -> more aggressive merging.
    single_tube    : bool   if True, keep ONLY the single best candidate
                            (largest area x SAM quality score). For the clinical
                            one-tube-per-image case: sidesteps fragments, shadow
                            blobs, and dedup entirely -- only one mask survives.
    debug          : bool   if True, print per-mask geometry for tuning.

    Returns
    -------
    list of dicts: bbox (x,y,w,h), mask (bool), area (int), score (float)
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)

    H, W = img.shape[:2]
    img_area = H * W

    if debug:
        print(f"[detect_tubes] SAM returned {len(masks)} raw masks")

    candidates = []
    for idx, m in enumerate(masks):
        seg = m["segmentation"].astype(np.uint8)
        area = int(seg.sum())

        # SAM's own quality estimate, used for dedup scoring.
        score = float(m.get("predicted_iou", 1.0)) * float(m.get("stability_score", 1.0))

        reject = None
        x, y, w, h = cv2.boundingRect(seg)
        aspect_ratio = h / w if w > 0 else 0

        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
        else:
            solidity = extent = 0

        if area < area_min:
            reject = "area<min"
        elif area > img_area * max_area_fraction:
            reject = "background"
        elif aspect_ratio <= aspect_ratio_min:
            reject = "too wide"
        elif aspect_ratio >= aspect_ratio_max:
            reject = "too thin (sliver/shadow)"
        elif solidity <= solidity_min:
            reject = "low solidity"
        elif extent <= extent_min:
            reject = "low extent"

        if debug:
            print(f"  mask {idx:2d}: area={area:6d} ar={aspect_ratio:4.2f} "
                  f"sol={solidity:4.2f} ext={extent:4.2f} score={score:4.2f} "
                  f"-> {reject or 'KEEP'}")

        if reject is None:
            candidates.append({
                "bbox": (x, y, w, h),
                "mask": seg.astype(bool),
                "area": area,
                "score": score,
            })

    if debug:
        print(f"[detect_tubes] {len(candidates)} passed shape filter; deduping...")

    if single_tube:
        # Clinical one-tube-per-image case: skip dedup, just keep the single
        # best candidate. Largest area x quality favors a complete, confident
        # tube mask over fragments or shadow blobs.
        if not candidates:
            if debug:
                print("[detect_tubes] single_tube: no candidates passed filter")
            return []
        best = max(candidates, key=lambda c: c["area"] * c["score"])
        if debug:
            print(f"[detect_tubes] single_tube: kept best of {len(candidates)} "
                  f"(area={best['area']}, score={best['score']:.2f})")
        return [best]

    tubes = _dedup_masks(candidates, overlap_thresh=overlap_thresh)

    if debug:
        print(f"[detect_tubes] {len(tubes)} tubes after dedup")

    tubes.sort(key=lambda t: t["bbox"][0])
    return tubes


def draw_detections(img_rgb, tubes):
    """Overlay each tube mask + bounding box for visual inspection."""
    debug = img_rgb.copy()
    overlay = debug.copy()
    rng = np.random.default_rng(42)

    for i, t in enumerate(tubes):
        color = tuple(int(c) for c in rng.integers(60, 230, size=3))
        overlay[t["mask"]] = color
        x, y, w, h = t["bbox"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(debug, f"T{i+1}", (x + 2, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    blended = cv2.addWeighted(overlay, 0.4, debug, 0.6, 0)
    return blended