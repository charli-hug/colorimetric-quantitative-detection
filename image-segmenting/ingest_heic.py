"""
ingest_heic.py -- prepare iPhone HEIC images for the pipeline.

For each .HEIC/.heic in an input folder:
  1. reads the concentration from the EXIF ImageDescription field
     (what you typed in the iPhone Photos "caption"/description),
  2. converts the image to JPEG (the format detector_sam/pellet_extractor read),
  3. writes a concentrations.csv mapping the OUTPUT jpeg name -> concentration.

The concentration is parsed into both its raw text ("10 nM") and a molar float
(1e-8) so it is numeric and sortable for the dose-response regression. Negative
/ zero controls ("0", "negative", "neg", "control", blank) map to molar = 0.

Usage:
    python3 ingest_heic.py <input_heic_dir> <output_jpeg_dir>
    # e.g.
    python3 ingest_heic.py concentration-heic concentration-images
"""
import os
import re
import sys
import csv

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

UNIT_SCALE = {
    "M": 1.0, "mM": 1e-3,
    "uM": 1e-6, "µM": 1e-6, "μM": 1e-6,
    "nM": 1e-9, "pM": 1e-12, "fM": 1e-15,
}


def parse_concentration(text):
    """
    '10 nM' -> {'raw':'10 nM','value':10.0,'unit':'nM','molar':1e-8}
    '0' / 'negative' / '' -> {'raw':<text>,'molar':0.0}  (zero control)
    Unrecognized -> {'raw':<text>,'molar':None}
    """
    if text is None:
        return {"raw": "", "molar": 0.0, "note": "blank -> treated as control"}
    t = str(text).strip()
    low = t.lower()
    if low in ("", "0", "negative", "neg", "control", "blank", "ntc", "0 m", "0m"):
        return {"raw": t, "molar": 0.0, "note": "control/zero"}
    m = re.match(r"([\d.]+)\s*([pfnPFNuUµμmM]?[mM])\b", t)
    if not m:
        # fallback: looser match on any 1-2 letter unit token
        m = re.match(r"([\d.]+)\s*([A-Za-zµμ]{1,2})", t)
        if not m:
            return {"raw": t, "molar": None, "note": "unparsed"}
    val = float(m.group(1))
    unit = m.group(2)
    # normalise unit capitalisation (nm -> nM, etc.)
    unit_norm = None
    for u in UNIT_SCALE:
        if unit.lower() == u.lower():
            unit_norm = u
            break
    if unit_norm is None:
        return {"raw": t, "molar": None, "note": f"unknown unit '{unit}'"}
    return {"raw": t, "value": val, "unit": unit_norm,
            "molar": val * UNIT_SCALE[unit_norm], "note": "ok"}


def read_description(img):
    """Return the EXIF ImageDescription text, or None."""
    try:
        exif = img.getexif()
        # 0x010E is ImageDescription
        desc = exif.get(0x010E)
        if desc and str(desc).strip():
            return str(desc).strip()
    except Exception:
        pass
    return None


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    heics = sorted(f for f in os.listdir(in_dir)
                   if f.lower().endswith((".heic", ".heif")))
    if not heics:
        print(f"No HEIC files found in {in_dir}")
        return

    rows = []
    print(f"Found {len(heics)} HEIC file(s).\n")
    for fn in heics:
        path = os.path.join(in_dir, fn)
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"  [SKIP] {fn}: cannot open ({e})")
            continue

        desc = read_description(img)
        conc = parse_concentration(desc)

        out_name = os.path.splitext(fn)[0] + ".jpeg"
        out_path = os.path.join(out_dir, out_name)
        img.convert("RGB").save(out_path, "JPEG", quality=95)

        molar = conc["molar"]
        molar_str = "" if molar is None else repr(molar)
        rows.append({
            "image": out_name,
            "source_heic": fn,
            "concentration_raw": conc["raw"],
            "concentration_molar": molar_str,
        })
        flag = "" if conc["note"] == "ok" or conc["note"].startswith("control") \
               else f"  <-- CHECK: {conc['note']}"
        print(f"  {fn}  ->  {out_name}   [{conc['raw'] or '(blank)'}"
              f" = {molar_str or 'NA'} M]{flag}")

    csv_path = os.path.join(out_dir, "concentrations.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "source_heic",
                                          "concentration_raw",
                                          "concentration_molar"])
        w.writeheader()
        w.writerows(rows)

    # quick summary of the series found
    seen = {}
    for r in rows:
        seen.setdefault(r["concentration_raw"] or "(blank)", 0)
        seen[r["concentration_raw"] or "(blank)"] += 1
    print(f"\nWrote {len(rows)} JPEG(s) and {csv_path}")
    print("Concentration series found (label: count):")
    for k, v in sorted(seen.items()):
        print(f"   {k:>12} : {v}")
    unparsed = [r["image"] for r in rows if r["concentration_molar"] == ""]
    if unparsed:
        print("\nImages with unparsed/missing concentration (fix the EXIF "
              "caption or edit concentrations.csv by hand):")
        for u in unparsed:
            print(f"   {u}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])