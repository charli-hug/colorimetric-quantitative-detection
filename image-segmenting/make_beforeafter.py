"""
make_beforeafter.py -- poster before/after figures.

For each input image, runs the pipeline and saves a clean two-panel figure:
  LEFT  "Input photograph"      -- the raw tube photo
  RIGHT "Automated detection"   -- same photo with the SAM tube mask (green)
                                   and the measured liquid-pool ROI (red) overlaid
The measured a* value is printed on the right panel so the visual connects to
the number.

Usage:
    python3 make_beforeafter.py <input_dir_or_image> <output_dir>
    # examples
    python3 make_beforeafter.py single-tube-images/IMG_8032.jpeg beforeafter
    python3 make_beforeafter.py concentration-jpeg beforeafter   # whole folder

Pick your best two pairs from the output for Figure 2.
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import detector_sam
import pellet_extractor

MODEL_TYPE = "mobile_sam"          # fast; use "vit_b" for the higher-quality model
CKPT = "mobile_sam.pt"             # or "sam_vit_b_01ec64.pth" for vit_b


def overlay(img_bgr, mask, color_bgr, alpha):
    out = img_bgr.copy()
    tint = np.zeros_like(out); tint[mask] = color_bgr
    return cv2.addWeighted(out, 1.0, tint, alpha, 0)


def process_one(path, mask_gen, outdir):
    img = cv2.imread(path)
    if img is None:
        print(f"  [skip] cannot read {path}")
        return
    tubes = detector_sam.detect_tubes(img, mask_gen, single_tube=True)
    if not tubes:
        print(f"  [skip] no tube detected in {os.path.basename(path)}")
        return
    t = tubes[0]
    pr = pellet_extractor.extract_pellet(img, t["mask"], t["bbox"], readout="fluid")
    m = pellet_extractor.measure_pellet(img, pr["roi_mask"], space="all")
    a_val = m.get("LAB", {}).get("a", 0.0)

    # build the overlay: tube mask green, ROI red
    vis = overlay(img, t["mask"], (0, 200, 0), 0.28)     # green (BGR)
    vis = overlay(vis, pr["roi_mask"], (0, 0, 255), 0.55)  # red (BGR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4.2), dpi=200)
    ax[0].imshow(img_rgb); ax[0].set_title("Input photograph", fontsize=12)
    ax[1].imshow(vis_rgb)
    ax[1].set_title("Automated detection", fontsize=12)
    # annotate a* on the detection panel
    ax[1].text(0.03, 0.97, f"a* = {a_val:.1f}",
               transform=ax[1].transAxes, va="top", ha="left",
               fontsize=12, color="white", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.6))
    for a in ax:
        a.axis("off")
    # small legend for the overlay colors
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color="#00C800", label="tube (SAM)"),
                          Patch(color="#FF0000", label="measured pool")],
                 loc="lower right", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    base = os.path.splitext(os.path.basename(path))[0]
    outpath = os.path.join(outdir, f"{base}_beforeafter.png")
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {outpath}   (a* = {a_val:.1f})")


def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    mask_gen = detector_sam.load_sam(model_type=MODEL_TYPE, checkpoint=CKPT, device="cpu")
    if os.path.isdir(inp):
        files = sorted(os.path.join(inp, f) for f in os.listdir(inp)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    else:
        files = [inp]
    print(f"Processing {len(files)} image(s)...")
    for f in files:
        process_one(f, mask_gen, outdir)
    print(f"\nDone. Pick your best two pairs from {outdir}/ for Figure 2.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
