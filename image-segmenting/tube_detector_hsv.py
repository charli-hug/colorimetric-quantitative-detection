import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── 1. LOAD IMAGE ──────────────────────────────────────────────────────────────
img = cv2.imread("tubes.png")
assert img is not None, "Image not found — check your filename/path"
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for matplotlib display

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
        continue  # skip tiny noise contours

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / w
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    extent = area / (w * h)

    if aspect_ratio > 2.5 and solidity > 0.6 and extent > 0.4:
        tube_rois.append((x, y, w, h))
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"AR:{aspect_ratio:.1f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

print(f"Detected {len(tube_rois)} tube(s)")

# ── 5. TIP ROI EXTRACTION + BUBBLE MASKING ─────────────────────────────────────
results = []

for i, (x, y, w, h) in enumerate(tube_rois):

    # Trim tube wall edges inward
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.05)
    x1 = x + margin_x
    x2 = x + w - margin_x
    y1 = y + margin_y
    y2 = y + h - margin_y

    # Isolate conical tip (bottom 28% of tube)
    tip_y_start = y1 + int((y2 - y1) * 0.72)
    tip_roi = img[tip_y_start:y2, x1:x2]

    if tip_roi.size == 0:
        continue

    # Bubble masking in HSV
    hsv_roi = cv2.cvtColor(tip_roi, cv2.COLOR_BGR2HSV)
    bubble_mask = (hsv_roi[:, :, 1] < 30) | (hsv_roi[:, :, 2] > 230)
    valid_pixels = tip_roi[~bubble_mask]

    if len(valid_pixels) == 0:
        print(f"Tube {i+1}: no valid pixels after bubble masking")
        continue

    mean_bgr = valid_pixels.mean(axis=0)
    mean_rgb = mean_bgr[::-1]  # flip to RGB for reporting
    results.append({
        "tube": i + 1,
        "mean_rgb": mean_rgb,
        "valid_pixel_count": len(valid_pixels),
        "tip_coords": (x1, tip_y_start, x2, y2)
    })

    # Draw tip ROI on debug image
    cv2.rectangle(debug_img, (x1, tip_y_start), (x2, y2), (255, 100, 0), 2)
    cv2.putText(debug_img, f"T{i+1}", (x1, tip_y_start - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

# ── 6. OUTPUT ──────────────────────────────────────────────────────────────────
for r in results:
    rgb = r["mean_rgb"]
    print(f"Tube {r['tube']}: mean RGB = ({rgb[0]:.1f}, {rgb[1]:.1f}, {rgb[2]:.1f}) "
          f"| valid pixels = {r['valid_pixel_count']}")

# Visual output
plt.figure(figsize=(10, 7))
plt.imshow(debug_img)
plt.title("Detected tubes (green) and tip ROIs (orange)")
plt.axis("off")
plt.tight_layout()
plt.savefig("output_debug.png", dpi=150)
plt.show()