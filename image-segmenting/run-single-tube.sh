#!/usr/bin/env bash
#
# run_single_tube.sh -- one-command batch run for the single-tube image set.
# Reads from single-tube-images/, writes figures + measurements.csv to
# single-tube-results/, runs in single-tube mode (keeps one object per image).
#
# Usage:
#   bash run_single_tube.sh            # normal run
#   bash run_single_tube.sh --debug    # also print per-image detection stats

set -euo pipefail

IN_DIR="single-tube-images"
OUT_DIR="single-tube-results"
CSV="${OUT_DIR}/measurements.csv"

if [ ! -d "$IN_DIR" ]; then
  echo "Input directory '$IN_DIR' not found (run this from the project folder)." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$CSV"

# Forward any extra flags (e.g. --debug) to the runner.
if [ "$#" -gt 0 ]; then EXTRA=("$@"); else EXTRA=(); fi

echo "Input : $IN_DIR"
echo "Output: $OUT_DIR"
echo "CSV   : $CSV"
echo "Mode  : single-tube"
echo

shopt -s nullglob nocaseglob
images=("$IN_DIR"/*.jpg "$IN_DIR"/*.jpeg "$IN_DIR"/*.png)
shopt -u nullglob nocaseglob

if [ "${#images[@]}" -eq 0 ]; then
  echo "No images (.jpg/.jpeg/.png) found in '$IN_DIR'." >&2
  exit 1
fi

count=0
fail=0
for img in "${images[@]}"; do
  echo "=== $img ==="
  if python3 run_sam_analysis.py "$img" \
        --save --outdir "$OUT_DIR" --csv "$CSV" --single-tube \
        ${EXTRA[@]+"${EXTRA[@]}"}; then
    count=$((count + 1))
  else
    echo "  [FAILED] $img" >&2
    fail=$((fail + 1))
  fi
  echo
done

echo "------------------------------------------------------------"
echo "Done. Processed $count image(s), $fail failure(s)."
echo "Figures: $OUT_DIR/*_sam_analysis.png"
echo "Table  : $CSV"