"""
run_sam_analysis.py
-------------------
Full pipeline: SAM tube detection -> chroma-based pellet extraction ->
HSV + LAB color measurement, with a debug visualization.

Usage:
    python3 run_sam_analysis.py images/IMG_0969.jpeg
    python3 run_sam_analysis.py images/IMG_0969.jpeg --model vit_b --ckpt sam_vit_b_01ec64.pth
    python3 run_sam_analysis.py images/IMG_0969.jpeg --debug   # print per-mask stats

Make sure your SAM weight file is in the project folder (see detector_sam.py).
"""

import sys
import os
import csv
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import detector_sam
import pellet_extractor
import preprocess


parser = argparse.ArgumentParser(description="PCR tube + pellet analysis (SAM)")
parser.add_argument("image", help="Path to input image")
parser.add_argument("--model", default="vit_b",
                    choices=["mobile_sam", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--ckpt", default="sam_vit_b_01ec64.pth",
                    help="Path to SAM checkpoint file")
parser.add_argument("--overlap-thresh", type=float, default=0.5,
                    help="Dedup aggressiveness: lower merges more (0-1)")
parser.add_argument("--debug", action="store_true",
                    help="Print per-mask geometry from SAM for tuning")
parser.add_argument("--pre-mask", dest="pre_mask", action="store_true",
                    help="GrabCut background/shadow suppression before SAM. "
                         "OFF by default -- it can clip tube bottoms and split "
                         "overlapping clusters, damaging the measurement region. "
                         "Only helps on clean, well-separated, shadow-heavy "
                         "backgrounds (e.g. the gray-table single-row shots).")
parser.set_defaults(pre_mask=False)
parser.add_argument("--single-tube", dest="single_tube", action="store_true",
                    help="Keep only the single best tube per image. Use for the "
                         "clinical one-tube-per-image case -- avoids fragments, "
                         "shadow masks, and overlap entirely.")
parser.add_argument("--csv", default=None,
                    help="Append per-tube measurements to this CSV file "
                         "(written if absent). Lets batch runs accumulate one "
                         "comparable table across all images.")
parser.add_argument("--save", action="store_true")
parser.add_argument("--outdir", default=None,
                    help="Directory to write the saved result figure into "
                         "(used with --save). Defaults to alongside the input.")
args = parser.parse_args()


# -- LOAD IMAGE --------------------------------------------------------------
img = cv2.imread(args.image)
if img is None:
    print(f"Error: could not open '{args.image}'.")
    sys.exit(1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Loaded {args.image}  ({img.shape[1]}x{img.shape[0]})")

# Optional background/shadow suppression. SAM runs on the masked image so
# shadows can't bleed into tube masks; ALL color measurement still uses the
# untouched original `img`.
img_for_sam = img
fg_mask = None
if args.pre_mask:
    print("Pre-masking background with GrabCut...")
    img_for_sam, fg_mask = preprocess.suppress_background(img)


# -- LOAD SAM + DETECT -------------------------------------------------------
print("Loading SAM (first load is the slow part)...")
mask_gen = detector_sam.load_sam(model_type=args.model, checkpoint=args.ckpt, device="cpu")

print("Segmenting...")
tubes = detector_sam.detect_tubes(img_for_sam, mask_gen,
                                  overlap_thresh=args.overlap_thresh,
                                  single_tube=args.single_tube,
                                  debug=args.debug)
print(f"Detected {len(tubes)} tube(s)")

if not tubes:
    print("No tubes found. Try lowering area_min/aspect_ratio_min in detector_sam.py,")
    print("or re-run with --debug to see why masks are being rejected.")
    sys.exit(0)


# -- EXTRACT + MEASURE PELLETS -----------------------------------------------
pellet_results = []
measurements = []
csv_rows = []

for i, t in enumerate(tubes):
    pr = pellet_extractor.extract_pellet(img, t["mask"], t["bbox"])
    pellet_results.append(pr)

    m = pellet_extractor.measure_pellet(img, pr["valid_mask"], space="all")
    measurements.append(m)

    lab = m.get("LAB", {})
    hsv = m.get("HSV", {})
    rgb = m.get("RGB", {})
    flag = "" if pr["reliable"] else "  [!] small ROI"
    print(f"  T{i+1}: ROI px={pr['pixel_count']}{flag} | "
          f"L*={lab.get('L',0):.1f} a*={lab.get('a',0):.1f} b*={lab.get('b',0):.1f} | "
          f"H={hsv.get('H',0):.1f} S={hsv.get('S',0):.1f} V={hsv.get('V',0):.1f} | "
          f"redDom={m.get('red_dominance',0):.1f} invMGV={m.get('inv_mgv',0):.1f}")

    csv_rows.append({
        "image": os.path.basename(args.image),
        "tube": i + 1,
        "roi_px": pr["pixel_count"],
        "reliable": pr["reliable"],
        "L": round(lab.get("L", 0), 3),
        "a": round(lab.get("a", 0), 3),
        "b": round(lab.get("b", 0), 3),
        "H": round(hsv.get("H", 0), 3),
        "S": round(hsv.get("S", 0), 3),
        "V": round(hsv.get("V", 0), 3),
        "R": round(rgb.get("R", 0), 3),
        "G": round(rgb.get("G", 0), 3),
        "B": round(rgb.get("B", 0), 3),
        "red_dominance": round(m.get("red_dominance", 0), 3),
        "gray_mean": round(m.get("gray_mean", 0), 3),
        "inv_mgv": round(m.get("inv_mgv", 0), 3),
    })

# Append (or create) the shared CSV so batch runs accumulate one table.
if args.csv and csv_rows:
    fields = list(csv_rows[0].keys())
    need_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if need_header:
            writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  -> appended {len(csv_rows)} rows to {args.csv}")


# -- VISUALIZE ---------------------------------------------------------------
# Build the panel list dynamically so the GrabCut step only appears when used.
det_vis = detector_sam.draw_detections(img_rgb, tubes)
pel_vis = pellet_extractor.draw_pellets(img_rgb, tubes, pellet_results)

panels = [(img_rgb, "1. Original")]

if args.pre_mask and fg_mask is not None:
    grabcut_rgb = cv2.cvtColor(img_for_sam, cv2.COLOR_BGR2RGB)
    panels.append((grabcut_rgb, "2. GrabCut foreground\n(bg+shadow removed)"))

panels.append((det_vis, f"{len(panels)+1}. SAM tubes ({len(tubes)})"))
panels.append((pel_vis, f"{len(panels)+1}. Liquid-pool ROI"))

# Final panel: a contact strip of just the ROI pixels per tube, so you can see
# exactly which pixels are being measured (the actual color samples).
roi_strip = img_rgb.copy()
dark = (roi_strip * 0.18).astype(roi_strip.dtype)   # dim everything...
for pr in pellet_results:
    if pr is not None:
        dark[pr["pellet_mask"]] = roi_strip[pr["pellet_mask"]]  # ...except ROIs
panels.append((dark, f"{len(panels)+1}. Measured pixels only"))

n = len(panels)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
if n == 1:
    axes = [axes]
for ax, (im, title) in zip(axes, panels):
    ax.imshow(im)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.suptitle(f"SAM pipeline: {args.image}"
             f"{'  [pre-masked]' if args.pre_mask else ''}", fontsize=11)
plt.tight_layout()

if args.save:
    base = os.path.splitext(os.path.basename(args.image))[0] + "_sam_analysis.png"
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, base)
    else:
        out = args.image.rsplit(".", 1)[0] + "_sam_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)        # don't block batch runs with an open window
else:
    plt.show()