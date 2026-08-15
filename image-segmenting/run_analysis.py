"""
run_analysis.py
---------------
Main entry point. Runs tube detection once, then passes each tube's
tip ROI through both the HSV and CIELAB processors for comparison.

Usage:
    python3 run_analysis.py tubes.png
    python3 run_analysis.py tubes.png --method hsv
    python3 run_analysis.py tubes.png --method lab
"""

import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tube_detector
import processor_hsv
import processor_cielab


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="PCR tube color analysis")
parser.add_argument("image", help="Path to input image")
parser.add_argument(
    "--method",
    choices=["hsv", "lab", "both"],
    default="both",
    help="Color method to run (default: both)"
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Save debug images to disk"
)
args = parser.parse_args()


# ── LOAD ──────────────────────────────────────────────────────────────────────
img = cv2.imread(args.image)
if img is None:
    print(f"Error: could not open '{args.image}'. Check the path.")
    sys.exit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"\nLoaded: {args.image}  ({img.shape[1]}×{img.shape[0]} px)")


# ── DETECTION (runs once, shared by both methods) ─────────────────────────────
print("\n── Detection ──────────────────────────────────────────────────────────")
tubes = tube_detector.detect_tubes(img)
print(f"Found {len(tubes)} tube(s)")

if len(tubes) == 0:
    print("No tubes detected. Try adjusting thresholds in tube_detector.py.")
    sys.exit(0)


# ── MEASUREMENT ───────────────────────────────────────────────────────────────
run_hsv = args.method in ("hsv", "both")
run_lab = args.method in ("lab", "both")

hsv_results = []
lab_results = []

for i, tube in enumerate(tubes):
    tip = tube["tip_coords"]
    label = f"Tube {i + 1}"

    print(f"\n{label}:")

    if run_hsv:
        r = processor_hsv.measure(img, tip)
        if r:
            hsv_results.append(r)
            print(f"  {processor_hsv.format_result(r)}")
        else:
            print(f"  [HSV] No valid pixels after masking")
            hsv_results.append(None)

    if run_lab:
        r = processor_lab.measure(img, tip)
        if r:
            lab_results.append(r)
            print(f"  {processor_lab.format_result(r)}")
        else:
            print(f"  [LAB] No valid pixels after masking")
            lab_results.append(None)


# ── DELTA-E (LAB only, between consecutive tubes) ─────────────────────────────
if run_lab and len([r for r in lab_results if r]) > 1:
    valid = [(i, r) for i, r in enumerate(lab_results) if r]
    print("\n── ΔE between tubes (CIELAB) ──────────────────────────────────────────")
    ref_i, ref = valid[0]
    for i, r in valid[1:]:
        dE = processor_lab.delta_e(ref, r)
        print(f"  Tube {ref_i+1} → Tube {i+1}: ΔE = {dE:.3f}")


# ── VISUALISATION ─────────────────────────────────────────────────────────────
debug_img = tube_detector.draw_detections(img_rgb, tubes)

n_plots = 1 + int(run_hsv) + int(run_lab)
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 7))
if n_plots == 1:
    axes = [axes]

ax_idx = 0

# Detection overlay
axes[ax_idx].imshow(debug_img)
axes[ax_idx].set_title("Detection\n(green=tube, orange=tip ROI)")
axes[ax_idx].axis("off")
ax_idx += 1

# HSV bar chart
if run_hsv and hsv_results:
    valid = [(i, r) for i, r in enumerate(hsv_results) if r]
    labels = [f"T{i+1}" for i, _ in valid]
    H = [r["H"] for _, r in valid]
    S = [r["S"] for _, r in valid]
    V = [r["V"] for _, r in valid]
    x = np.arange(len(labels))
    width = 0.25
    ax = axes[ax_idx]
    ax.bar(x - width, H, width, label="H (×1)", color="tomato")
    ax.bar(x,         S, width, label="S",       color="steelblue")
    ax.bar(x + width, V, width, label="V",       color="gold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 255)
    ax.set_title("HSV measurements")
    ax.legend()
    ax_idx += 1

# LAB bar chart
if run_lab and lab_results:
    valid = [(i, r) for i, r in enumerate(lab_results) if r]
    labels = [f"T{i+1}" for i, _ in valid]
    L = [r["L"] for _, r in valid]
    a = [r["a"] for _, r in valid]
    b = [r["b"] for _, r in valid]
    x = np.arange(len(labels))
    width = 0.25
    ax = axes[ax_idx]
    ax.bar(x - width, L, width, label="L*",  color="lightgray", edgecolor="black")
    ax.bar(x,         a, width, label="a*",  color="tomato")
    ax.bar(x + width, b, width, label="b*",  color="gold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("CIELAB measurements")
    ax.legend()
    ax_idx += 1

plt.suptitle(f"PCR Tube Color Analysis — {args.image}", fontsize=11)
plt.tight_layout()

if args.save:
    out_path = args.image.rsplit(".", 1)[0] + "_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved debug image: {out_path}")

plt.show()