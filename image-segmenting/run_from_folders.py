"""
run_from_folders.py -- batch pipeline run with labels from subfolder name.

Expects:
    final-concentration-images/
        high/       <- tube photos, this concentration tier
        low/
        negative/

Runs the same detection + ROI + measurement pipeline as run_sam_analysis.py on
every image, and writes:
    measurements.csv      -- same schema as the existing pipeline (image, tube,
                              readout, roi_px, reliable, L, a, b, H, S, V, R, G, B,
                              red_dominance, gray_mean, inv_mgv, bg_R, bg_G, bg_B)
    concentrations.csv     -- image, concentration_raw (= folder name),
                              concentration_molar (placeholder numeric so the
                              existing tier scripts -- sixpanel_tiers.R,
                              stress_test.R -- work UNCHANGED: negative->0,
                              low->1e-9, high->1e-7. Only the >=1e-7 / ==0
                              boundaries matter to those scripts, so the exact
                              placeholder values don't affect the tier grouping.)

Subfolder names are matched case-insensitively; "neg" also accepted for negative.

Usage:
    python3 run_from_folders.py final-concentration-images final-concentration-results
"""
import os
import sys
import csv
import cv2
import numpy as np

import detector_sam
import pellet_extractor

MODEL_TYPE = "vit_b"                    # use "mobile_sam" for faster/lower-quality
CKPT = "sam_vit_b_01ec64.pth"           # matching checkpoint

TIER_FOLDER_MAP = {
    "high": ("high", 1e-7),
    "low": ("low", 1e-9),
    "negative": ("negative", 0.0),
    "neg": ("negative", 0.0),
}

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_image(path):
    """Read a standard image; falls back to HEIC via pillow-heif if needed."""
    if path.lower().endswith((".heic", ".heif")):
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)[:, :, ::-1].copy()   # RGB -> BGR for OpenCV
        return arr
    return cv2.imread(path)


def process_one(path, tier_name, mask_gen, csv_rows):
    img = load_image(path)
    if img is None:
        print(f"  [skip] cannot read {path}")
        return

    tubes = detector_sam.detect_tubes(img, mask_gen, single_tube=True)
    if not tubes:
        print(f"  [skip] no tube detected: {os.path.basename(path)}")
        return
    t = tubes[0]

    pr = pellet_extractor.extract_pellet(img, t["mask"], t["bbox"], readout="fluid")
    m = pellet_extractor.measure_pellet(img, pr["roi_mask"], space="all")
    lab = m.get("LAB", {}); hsv = m.get("HSV", {}); rgb = m.get("RGB", {})

    # per-image background illuminant (tube-free patch), same logic as run_sam_analysis.py
    H, W = img.shape[:2]
    dil = cv2.dilate(t["mask"].astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)), 1).astype(bool)
    bg_mask = ~dil
    if bg_mask.sum() > 100:
        bg_bgr = img[bg_mask].mean(axis=0)
        bg_R, bg_G, bg_B = float(bg_bgr[2]), float(bg_bgr[1]), float(bg_bgr[0])
    else:
        bg_R = bg_G = bg_B = float("nan")

    fname = os.path.basename(path)
    csv_rows.append({
        "image": fname, "tube": 1, "readout": "fluid",
        "roi_px": pr["pixel_count"], "reliable": pr["reliable"],
        "L": round(lab.get("L", 0), 3), "a": round(lab.get("a", 0), 3),
        "b": round(lab.get("b", 0), 3),
        "H": round(hsv.get("H", 0), 3), "S": round(hsv.get("S", 0), 3),
        "V": round(hsv.get("V", 0), 3),
        "R": round(rgb.get("R", 0), 3), "G": round(rgb.get("G", 0), 3),
        "B": round(rgb.get("B", 0), 3),
        "red_dominance": round(m.get("red_dominance", 0), 3),
        "gray_mean": round(m.get("gray_mean", 0), 3),
        "inv_mgv": round(m.get("inv_mgv", 0), 3),
        "bg_R": round(bg_R, 3), "bg_G": round(bg_G, 3), "bg_B": round(bg_B, 3),
        "_tier": tier_name,
    })
    print(f"  {fname:20} tier={tier_name:9} a*={lab.get('a',0):+.2f}")


def main(root, outdir):
    os.makedirs(outdir, exist_ok=True)
    mask_gen = detector_sam.load_sam(model_type=MODEL_TYPE, checkpoint=CKPT, device="cpu")

    csv_rows = []
    found_any_folder = False
    for sub in sorted(os.listdir(root)):
        subpath = os.path.join(root, sub)
        if not os.path.isdir(subpath):
            continue
        key = sub.strip().lower()
        if key not in TIER_FOLDER_MAP:
            print(f"[warn] folder '{sub}' doesn't match high/low/negative -- skipping")
            continue
        found_any_folder = True
        tier_name, _ = TIER_FOLDER_MAP[key]
        files = sorted(f for f in os.listdir(subpath)
                       if f.lower().endswith(IMG_EXTS + (".heic", ".heif")))
        print(f"\n=== {sub}/  ({len(files)} image(s), tier={tier_name}) ===")
        for f in files:
            process_one(os.path.join(subpath, f), tier_name, mask_gen, csv_rows)

    if not found_any_folder:
        print(f"No high/low/negative subfolders found under {root}. "
              f"Expected {root}/high, {root}/low, {root}/negative.")
        return

    # measurements.csv (drop internal _tier column from this file)
    mcsv_path = os.path.join(outdir, "measurements.csv")
    fieldnames = ["image", "tube", "readout", "roi_px", "reliable", "L", "a", "b",
                 "H", "S", "V", "R", "G", "B", "red_dominance", "gray_mean",
                 "inv_mgv", "bg_R", "bg_G", "bg_B"]
    with open(mcsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: row[k] for k in fieldnames})

    # concentrations.csv (tier-derived, compatible with sixpanel_tiers.R / stress_test.R)
    ccsv_path = os.path.join(outdir, "concentrations.csv")
    with open(ccsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "concentration_raw", "concentration_molar"])
        w.writeheader()
        for row in csv_rows:
            _, molar = TIER_FOLDER_MAP[row["_tier"]] if row["_tier"] in ("high","low","negative") \
                       else (row["_tier"], None)
            # row["_tier"] is already the normalized tier name; map back to molar directly
            molar = {"high": 1e-7, "low": 1e-9, "negative": 0.0}[row["_tier"]]
            w.writerow({"image": row["image"], "concentration_raw": row["_tier"],
                       "concentration_molar": repr(molar)})

    n_by_tier = {}
    for row in csv_rows:
        n_by_tier[row["_tier"]] = n_by_tier.get(row["_tier"], 0) + 1
    print(f"\nDone. {len(csv_rows)} tube(s) processed: {n_by_tier}")
    print(f"Wrote {mcsv_path}")
    print(f"Wrote {ccsv_path}")

    run_r_figures(mcsv_path, ccsv_path, outdir)


def run_r_figures(mcsv_path, ccsv_path, outdir):
    """Run the tier figure scripts against the just-written CSVs, if present."""
    import subprocess, shutil
    if shutil.which("Rscript") is None:
        print("\n[skip] Rscript not found on PATH -- run the R scripts manually:")
        print(f"  Rscript sixpanel_tiers.R {mcsv_path} {ccsv_path} {outdir}")
        print(f"  Rscript stress_test.R {mcsv_path} {ccsv_path} {outdir}")
        return

    here = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        ("sixpanel_tiers.R", "Figure 3 (six-panel channel comparison by tier)"),
        ("stress_test.R", "Figure 4/5 (lighting robustness + normalization)"),
    ]
    for script, desc in scripts:
        script_path = os.path.join(here, script)
        if not os.path.exists(script_path):
            print(f"\n[skip] {script} not found next to run_from_folders.py -- "
                 f"run it manually against {mcsv_path} / {ccsv_path}")
            continue
        print(f"\n=== Running {script}  ({desc}) ===")
        result = subprocess.run(
            ["Rscript", script_path, mcsv_path, ccsv_path, outdir],
            capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"[warning] {script} exited with an error:")
            print(result.stderr)
        else:
            print(f"{script} completed -> figures written to {outdir}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
