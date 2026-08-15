"""
tube_detector.py
----------------
Shared detection pipeline: background separation, contour finding,
shape filtering, and tip ROI extraction.

Returns a list of tube ROI dicts consumed by each processor.
"""

import cv2
import numpy as np


def detect_tubes(img, area_min=3000, aspect_ratio_min=1.8,
                 solidity_min=0.5, extent_min=0.35,
                 wall_margin_x=0.15, wall_margin_y=0.05,
                 tip_fraction=0.28):

    # ── PREPROCESSING ─────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold handles variable backgrounds much better than GrabCut
    # for closely-spaced objects
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=4
    )

    # Clean up small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Close small gaps between nearby edges so each tube stays as one blob
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # ── CONTOURS ──────────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── SHAPE FILTERING ───────────────────────────────────────────────────────
    tubes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / w
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        extent = area / (w * h)

        if (aspect_ratio > aspect_ratio_min
                and solidity > solidity_min
                and extent > extent_min):

            mx = int(w * wall_margin_x)
            my = int(h * wall_margin_y)
            x1 = x + mx
            x2 = x + w - mx
            y1 = y + my
            y2 = y + h - my

            tip_y_start = y1 + int((y2 - y1) * (1 - tip_fraction))

            tubes.append({
                "bbox": (x, y, w, h),
                "tip_coords": (x1, tip_y_start, x2, y2),
            })

    return tubes


def draw_detections(img_rgb, tubes):
    """
    Return a copy of img_rgb with bounding boxes and tip ROIs drawn on it.
    Green = full tube bbox, Orange = tip ROI.
    """
    debug = img_rgb.copy()
    for i, t in enumerate(tubes):
        x, y, w, h = t["bbox"]
        x1, ty1, x2, ty2 = t["tip_coords"]

        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug, f"T{i+1}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.rectangle(debug, (x1, ty1), (x2, ty2), (255, 100, 0), 2)

    return debug