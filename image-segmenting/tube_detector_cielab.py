import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── 1. LOAD IMAGE ──────────────────────────────────────────────────────────────
img = cv2.imread("tubes.png")
assert img is not None, "Image not found — check filename/path"
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ── 2. BACKGROUND SEPARATION (GrabCut) ─────────────────────────────────────────
mask = np.zeros(img.shape[:2], np.uint8)
rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
foreground = img * fg_mask[:, :, np.newaxis]

# ── 3. EDGE DETECTION + CONTOURS ───────────────────────────────────────────────
gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ── 4. SHAPE FILTERING ─────────────────────────────────────────────────────────
tube_rois = []
debug_img = img_rgb.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 3000:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / w
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    extent = area / (w * h)

    if aspect_ratio > 2.5 and solidity > 0.6 and extent > 0.4:
        tube_rois.append((x, y, w, h))
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f"Detected {len(tube_rois)} tube(s)")

# ── 5. CONVERT FULL IMAGE TO LAB ONCE (efficient) ──────────────────────────────
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

# ── 6. TIP ROI EXTRACTION ──────────────────────────────────────────────────────
results = []

for i, (x, y, w, h) in enumerate(tube_rois):

    # Trim tube wall edges inward
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.05)
    x1 = x + margin_x
    x2 = x + w - margin_x
    y1 = y + margin_y
    y2 = y + h - margin_y

    # Isolate conical tip (bottom 28%)
    tip_y_start = y1 + int((y2 - y1) * 0.72)

    # Extract tip region in both HSV (for masking) and LAB (for measurement)
    tip_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[tip_y_start:y2, x1:x2]
    tip_lab = img_lab[tip_y_start:y2, x1:x2]

    if tip_hsv.size == 0:
        continue

    # ── ARTIFACT MASKING (HSV) ─────────────────────────────────────────────────
    # Exclude bubbles, highlights, shadows using HSV — same logic as before
    bubble_mask = (tip_hsv[:, :, 1] < 30) | (tip_hsv[:, :, 2] > 230)
    shadow_mask = tip_hsv[:, :, 2] < 30
    artifact_mask = bubble_mask | shadow_mask
    valid_mask = ~artifact_mask

    valid_lab_pixels = tip_lab[valid_mask]

    if len(valid_lab_pixels) == 0:
        print(f"Tube {i+1}: no valid pixels after masking")
        continue

    # ── LAB MEASUREMENT ────────────────────────────────────────────────────────
    # OpenCV stores LAB as: L* in [0,255], a* and b* offset to [0,255]
    # Convert back to true LAB ranges: L* [0,100], a* [-128,127], b* [-128,127]
    mean_lab_raw = valid_lab_pixels.mean(axis=0)
    L = mean_lab_raw[0] * (100 / 255)
    a = mean_lab_raw[1] - 128
    b = mean_lab_raw[2] - 128

    results.append({
        "tube": i + 1,
        "L": L, "a": a, "b": b,
        "valid_pixel_count": len(valid_lab_pixels),
        "tip_coords": (x1, tip_y_start, x2, y2)
    })

    cv2.rectangle(debug_img, (x1, tip_y_start), (x2, y2), (255, 100, 0), 2)
    cv2.putText(debug_img, f"T{i+1}", (x1, tip_y_start - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

# ── 7. OUTPUT ──────────────────────────────────────────────────────────────────
for r in results:
    print(f"Tube {r['tube']}: L*={r['L']:.2f}  a*={r['a']:.2f}  b*={r['b']:.2f} "
          f"| valid pixels = {r['valid_pixel_count']}")

# Optional: compute deltaE between tube 1 and others as a difference metric
if len(results) > 1:
    ref = results[0]
    print("\n── ΔE from Tube 1 ──")
    for r in results[1:]:
        dE = np.sqrt((r['L']-ref['L'])**2 + (r['a']-ref['a'])**2 + (r['b']-ref['b'])**2)
        print(f"  Tube {r['tube']}: ΔE = {dE:.3f}")

plt.figure(figsize=(10, 7))
plt.imshow(debug_img)
plt.title("Detected tubes (green) and tip ROIs (orange) — LAB version")
plt.axis("off")
plt.tight_layout()
plt.savefig("output_debug_lab.png", dpi=150)
plt.show()