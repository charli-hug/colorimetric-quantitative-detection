import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Pass image path as argument: python3 debug_detection.py images/IMG_0969.jpeg
path = sys.argv[1] if len(sys.argv) > 1 else "images/IMG_0964.jpeg"

img = cv2.imread(path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# ── Step 1: Edge map (background-agnostic) ─────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 20, 60)  # low thresholds to catch faint tube walls

# ── Step 2: Vertical projection — find columns with strong vertical edges ──
# Each tube wall = a vertical stripe of high edge density
# Sum edge pixels column by column
col_projection = edges.sum(axis=0).astype(float)
col_projection /= col_projection.max()  # normalize to 0-1

# Sum edge pixels row by row — helps find top/bottom of tube region
row_projection = edges.sum(axis=1).astype(float)
row_projection /= row_projection.max()

# ── Step 3: Find tube column bands via projection peaks ────────────────────
# Smooth the projection first to merge nearby peaks
from scipy.signal import find_peaks
import scipy.ndimage as ndi
smoothed_col = ndi.gaussian_filter1d(col_projection, sigma=15)

# Find peaks = tube wall positions
peaks, props = find_peaks(smoothed_col, height=0.1, distance=w // 20)

# ── Step 4: Find vertical extent of tubes via row projection ──────────────
smoothed_row = ndi.gaussian_filter1d(row_projection, sigma=10)
# Tube region = rows with above-threshold edge activity
row_thresh = 0.1
tube_rows = np.where(smoothed_row > row_thresh)[0]
if len(tube_rows) > 0:
    row_top    = max(0, tube_rows[0] - 20)
    row_bottom = min(h, tube_rows[-1] + 20)
else:
    row_top, row_bottom = 0, h

# ── Step 5: Find tubes as valleys between peaks (valley-to-valley) ─────────
# Invert projection to find valleys = gaps between tubes
inverted = 1 - smoothed_col

# Find valleys in the original = gaps between tubes
valleys, _ = find_peaks(inverted, height=0.5, distance=w // 15)

# Add image edges as boundaries
# Fix 1: don't use image edges as boundaries — only real valleys
# Remove the 0 and w from boundaries, filter out bands that are just margins
boundaries = valleys  # real valleys only, no image edges added

tube_bands = []
for i in range(len(boundaries) - 1):
    x1 = int(boundaries[i])
    x2 = int(boundaries[i + 1])
    band_w = x2 - x1
    band_h = row_bottom - row_top
    ar = band_h / band_w if band_w > 0 else 0

    # Must be tall enough and wide enough to be a real tube
    if ar > 1.5 and band_w > w // 15:
        tube_bands.append((x1, row_top, x2, row_bottom))

# Fix 2: compute tip ROI per tube using its own column's row projection
# instead of sharing one global row_bottom
tip_bands = []
for (x1, y1, x2, y2) in tube_bands:
    # Per-tube row projection using only columns within this band
    tube_col_edges = edges[:, x1:x2]
    tube_row_proj = tube_col_edges.sum(axis=1).astype(float)
    if tube_row_proj.max() > 0:
        tube_row_proj /= tube_row_proj.max()
    tube_row_smooth = ndi.gaussian_filter1d(tube_row_proj, sigma=8)

    # Find where this tube's edges actually end vertically
    active_rows = np.where(tube_row_smooth > 0.1)[0]
    if len(active_rows) > 0:
        t_top    = max(0, active_rows[0])
        t_bottom = min(h, active_rows[-1])
    else:
        t_top, t_bottom = y1, y2

    tip_y = t_top + int((t_bottom - t_top) * 0.75)
    tip_bands.append((x1, t_top, x2, t_bottom, tip_y))

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 10))

# Original + detected bands
ax1 = fig.add_subplot(2, 3, 1)
vis = img_rgb.copy()
for (x1, y1, x2, y2) in tube_bands:
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
ax1.imshow(vis)
ax1.set_title(f"Detected {len(tube_bands)} tube(s)")
ax1.axis("off")

# Edge map
ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(edges, cmap="gray")
ax2.set_title("Edge map")
ax2.axis("off")

# Column projection
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(smoothed_col, label="col projection")
ax3.plot(peaks, smoothed_col[peaks], "rx", markersize=8, label="peaks")
ax3.axhline(0.1, color="gray", linestyle="--", label="threshold")
ax3.set_title("Column projection\n(peaks = tube walls)")
ax3.legend(fontsize=8)

# Row projection
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(smoothed_row, label="row projection")
ax4.axhline(row_thresh, color="gray", linestyle="--")
ax4.axvline(row_top,    color="green", linestyle="--", label="tube top")
ax4.axvline(row_bottom, color="red",   linestyle="--", label="tube bottom")
ax4.set_title("Row projection\n(tube vertical extent)")
ax4.legend(fontsize=8)

# Individual tube ROIs
ax5 = fig.add_subplot(2, 3, 5)
# Replace the tip ROI drawing loop in the plot section:
rois_vis = img_rgb.copy()
for i, (x1, t_top, x2, t_bottom, tip_y) in enumerate(tip_bands):
    cv2.rectangle(rois_vis, (x1, t_top),  (x2, t_bottom), (0, 200, 0),   2)
    cv2.rectangle(rois_vis, (x1, tip_y),  (x2, t_bottom), (255, 100, 0), 2)
    cv2.putText(rois_vis, f"T{i+1}", (x1+2, t_top+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
ax5.imshow(rois_vis)
ax5.set_title("Tube bands (green)\nTip ROIs (orange)")
ax5.axis("off")

plt.suptitle(f"Projection-based detection: {path}", fontsize=10)
plt.tight_layout()
plt.savefig("debug_output.png", dpi=150)
plt.show()

print(f"\nDetected {len(tube_bands)} tube band(s)")
for i, (x1, y1, x2, y2) in enumerate(tube_bands):
    print(f"  T{i+1}: x={x1}–{x2}, y={y1}–{y2}, AR={((y2-y1)/(x2-x1)):.2f}")