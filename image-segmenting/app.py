"""
app.py -- ChromaShield colorimetric assay: proof-of-concept web demo.

A thin Streamlit front-end over the existing pipeline. Upload a tube photo,
the app segments the tube (MobileSAM), extracts the liquid-pool ROI, measures
its colour, and reports a positive/negative call from the a* channel.

RESEARCH DEMONSTRATION ONLY -- not a validated diagnostic device.

Run locally:
    streamlit run app.py

Requires (see SETUP.md):
    pip install streamlit opencv-python-headless numpy
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git
    # download mobile_sam.pt checkpoint into this folder
"""
import io
import os
import urllib.request
import numpy as np
import cv2
import streamlit as st

import detector_sam
import pellet_extractor

# ---- CONFIG ---------------------------------------------------------------
# The classification threshold from the labeled 28-image study (in-sample).
# EDIT THESE when you re-fit on new/validation data (see "changing the code").
# Three-tier a* cutoffs, from the concentration-series batches:
#   high (>=100nM): a* is far above everything else (100nM cluster ~13.8, huge
#     gap below it). HIGH_CUT sits in that gap.
#   low  (100pM-10nM): lifted above the negative band. LOW_CUT is the low-vs-
#     negative boundary (~ -0.23 from batch 2, 93% single-sample accuracy there).
#   negative: below LOW_CUT.
# The low/negative boundary is the UNCERTAIN one (batches disagreed on where low
# sits); BORDERLINE_MARGIN flags tubes close to it so the app hedges rather than
# overclaims. The high boundary is rock-solid.
HIGH_CUT = 7.0              # a* >= this -> high concentration (100nM range)
LOW_CUT  = -0.23           # a* >= this (and < HIGH_CUT) -> low; below -> negative
BORDERLINE_MARGIN = 1.0    # flag "borderline" within this a* distance of LOW_CUT
MOBILE_SAM_CKPT = "mobile_sam.pt"
MODEL_TYPE = "mobile_sam"   # uses the vit_t branch in detector_sam.load_sam

# If the checkpoint isn't present (e.g. fresh cloud deploy where the .pt isn't
# committed to git), download it once. Replace URL if this mirror changes.
MOBILE_SAM_URL = ("https://github.com/ChaoningZhang/MobileSAM/raw/master/"
                  "weights/mobile_sam.pt")


def ensure_checkpoint():
    if not os.path.exists(MOBILE_SAM_CKPT):
        with st.spinner("Downloading model weights (first run only)…"):
            urllib.request.urlretrieve(MOBILE_SAM_URL, MOBILE_SAM_CKPT)

st.set_page_config(page_title="Assay Demo", layout="wide")


# ---- MODEL (load once, cache across reruns) -------------------------------
@st.cache_resource
def get_mask_generator():
    ensure_checkpoint()
    return detector_sam.load_sam(model_type=MODEL_TYPE,
                                 checkpoint=MOBILE_SAM_CKPT, device="cpu")


def analyze(img_bgr):
    """Run the full pipeline on one BGR image. Returns a result dict or None."""
    mask_gen = get_mask_generator()
    tubes = detector_sam.detect_tubes(img_bgr, mask_gen, single_tube=True)
    if not tubes:
        return None
    t = tubes[0]
    pr = pellet_extractor.extract_pellet(img_bgr, t["mask"], t["bbox"],
                                         readout="fluid")
    m = pellet_extractor.measure_pellet(img_bgr, pr["roi_mask"], space="all")
    lab = m.get("LAB", {})
    return {
        "a": lab.get("a", 0.0),
        "L": lab.get("L", 0.0),
        "b": lab.get("b", 0.0),
        "measure": m,
        "roi_mask": pr["roi_mask"],
        "tube_mask": t["mask"],
        "roi_px": pr["pixel_count"],
        "reliable": pr["reliable"],
    }


def overlay(img_bgr, mask, color=(0, 0, 255), alpha=0.45):
    out = img_bgr.copy()
    tint = np.zeros_like(out); tint[mask] = color
    return cv2.addWeighted(out, 1.0, tint, alpha, 0)


# ---- UI -------------------------------------------------------------------
st.title("HPV Colorimetric Assay")
st.caption("Portable-diagnostics proof of concept — upload a tube photo to get "
           "an automated concentration-tier readout (high / low / negative) "
           "from the a\\* colour channel.")
st.warning("**Research demonstration only.** Not a validated diagnostic device. "
           "The classification threshold is fit in-sample on a small pilot set.")

uploaded = st.file_uploader("Upload a tube image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not read that image.")
        st.stop()

    with st.spinner("Segmenting and measuring…"):
        res = analyze(img_bgr)

    if res is None:
        st.error("No tube detected. Try a clearer, straight-on photo against a "
                 "plain background.")
        st.stop()

    a_val = res["a"]
    # Three-tier call from a*.
    if a_val >= HIGH_CUT:
        tier, tier_desc = "Positive - HIGH", "high concentration (~100 nM range)"
    elif a_val >= LOW_CUT:
        tier, tier_desc = "Positive - LOW", "low concentration (100 pM – 10 nM range)"
    else:
        tier, tier_desc = "NEGATIVE", "no target detected"
    near_lowneg = abs(a_val - LOW_CUT) < BORDERLINE_MARGIN

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Detected tube + measured region")
        vis = overlay(img_bgr, res["tube_mask"], (0, 200, 0), 0.25)
        vis = overlay(vis, res["roi_mask"], (0, 0, 255), 0.5)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Result")
        if tier == "Positive - HIGH":
            st.success(f"### Positive - HIGH\n{tier_desc}\n\na\\* = {a_val:.2f}  (≥ {HIGH_CUT})")
        elif tier == "Positive - LOW":
            st.warning(f"### Positive - LOW\n{tier_desc}\n\na\\* = {a_val:.2f}  "
                       f"({LOW_CUT} ≤ a\\* < {HIGH_CUT})", icon="🟡")
        else:
            st.info(f"### NEGATIVE\n{tier_desc}\n\na\\* = {a_val:.2f}  (< {LOW_CUT})")

        # Confidence signalling: the low/negative boundary is the uncertain one.
        if near_lowneg:
            st.warning("a\\* sits near the low/negative boundary — this call is "
                       "uncertain. The low-vs-negative distinction is the least "
                       "reliable part of the assay (~93% at single-tube level).",
                       icon="⚖️")
        if not res["reliable"]:
            st.warning("ROI is small — the measured region may not have captured "
                       "the pool well. Interpret with caution.", icon="🔍")

        st.markdown("**Colour channels**")
        st.table({
            "channel": ["a* (green–red)", "L* (lightness)", "b* (blue–yellow)",
                        "ROI pixels"],
            "value": [f"{a_val:.2f}", f"{res['L']:.1f}", f"{res['b']:.2f}",
                      f"{res['roi_px']}"],
        })
        st.caption("Higher a\\* = redder supernatant = more target. Tiers: "
                   "HIGH (~100 nM) is cleanly separated; LOW pools 100 pM–10 nM "
                   "(not resolved within); the LOW/NEGATIVE boundary is the "
                   "least certain. Thresholds from pilot batches; re-validate.")
else:
    st.info("Upload a tube photo to begin.")