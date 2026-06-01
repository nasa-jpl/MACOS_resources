#!/usr/bin/env bash
# run_dwdz.sh -- multi-field dw/dz_Zernike example for e5hex1.in
# (5-segment hex, used as a small-aperture cross-check).
#
# Inputs the user must set per Rx:
#   --field-x-rad / --field-y-rad : test-point inset from the
#       extreme corner of the science FoV.  e5hex1's prescription
#       ChfRayDir is (0, -3.49e-4, +1) -- i.e. already tilted ~70
#       µrad in y.  The corner-field offsets here are ADDITIVE on
#       top of that nominal direction (see field_to_chfraydir() in
#       dw_dz_zernike_multi.py).  100 µrad here is a placeholder;
#       replace with the FoV-matching value when you have it.
#
# See FFSegDemoAll/run_dwdz.sh for the full input/output catalogue.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/e5hex1.in"

echo "=== e5hex1: 5-field default ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dz_zernike_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --n-zcoef 5 \
    --field-x-rad 1e-4 \
    --field-y-rad 1e-4 \
    --out-dir "$SCRIPT_DIR"
