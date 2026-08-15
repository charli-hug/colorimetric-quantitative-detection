#!/usr/bin/env bash
#
# run_batch.sh -- run the SAM pellet pipeline over every image in a folder,
# collect the annotated result figures into an output directory, and append all
# per-tube measurements into a single CSV for direct comparison.
#
# Usage:
#   ./run_batch.sh                          # defaults: images/ -> results/
#   ./run_batch.sh images results           # custom input/output dirs
#   ./run_batch.sh images results --no-pre-mask   # pass extra flags to runner
#
# Any args after the 2nd are forwarded to run_sam_analysis.py (e.g.
# --no-pre-mask for clean white backgrounds, or --model/--ckpt overrides).

set -euo pipefail

IN_DIR="${1:-images}"
OUT_DIR="${2:-results}"
# Shift off the first two positional args (if present) so "$@" holds extras.
[ "$#" -ge 1 ] && shift || true
[ "$#" -ge 1 ] && shift || true
# Note: under macOS Bash 3.2 with `set -u`, expanding an empty array with
# "${arr[@]}" errors as "unbound". Guard every expansion with a length check.
EXTRA_ARGS=("$@")

CSV="${OUT_DIR}/measurements.csv"

if [ ! -d "$IN_DIR" ]; then
  echo "Input directory '$IN_DIR' not found." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Start each batch with a fresh CSV so reruns don't stack duplicate rows.
rm -f "$CSV"

echo "Input : $IN_DIR"
echo "Output: $OUT_DIR"
echo "CSV   : $CSV"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo "Extra : ${EXTRA_ARGS[*]}" || true
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
  # Build the arg list, appending extras only if any exist (Bash 3.2 safe).
  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    set_extra=("${EXTRA_ARGS[@]}")
  else
    set_extra=()
  fi
  if python3 run_sam_analysis.py "$img" \
        --save --outdir "$OUT_DIR" --csv "$CSV" \
        ${set_extra[@]+"${set_extra[@]}"}; then
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