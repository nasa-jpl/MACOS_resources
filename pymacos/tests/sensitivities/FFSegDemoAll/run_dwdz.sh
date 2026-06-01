#!/usr/bin/env bash
# run_dwdz.sh -- multi-field dw/dz_Zernike example for FFSegDemoAll.in.
#
# Two invocations:
#   1) default 5-field set (center + 4 corners) at ±field-x/y rad
#   2) explicit 3x3 grid -- exercises the NxM path and the de-dup
#      logic that ensures the center field is computed only once.
#
# Inputs that depend on the prescription (the user must set these):
#   --rx              path to the .in file
#   --field-x-rad     half-angle in x to place the corner test fields
#                     (50 µrad here -- generous for FFSegDemoAll's
#                     simple collimated source.  A real HWO-style FoV
#                     would be ~5-10 arcmin = ~1.5-3 mrad.)
#   --field-y-rad     half-angle in y (independent of x).
#
# Optional dev-speed knobs:
#   --model-size N    macos diffraction grid (default 128 here -- fast)
#   --n-zcoef K       number of Zernike sensitivity channels per
#                     element (default 5)
#
# Outputs land in this directory:
#   dwdzall_FFSegDemoAll.mat        -- dwdxall + w0_stacked + indxall
#   opdall_FFSegDemoAll.png         -- tiled nominal OPDs
#   opdall_diff_FFSegDemoAll.png    -- (per-field OPD) - (center OPD)
#
# Run from the pymacos repo root or from this directory.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/FFSegDemoAll.in"

echo "=== FFSegDemoAll: 5-field default ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dz_zernike_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --n-zcoef 5 \
    --field-x-rad 5e-5 \
    --field-y-rad 5e-5 \
    --tag FFSegDemoAll_5fp \
    --out-dir "$SCRIPT_DIR"

echo
echo "=== FFSegDemoAll: 3x3 grid (all FPs, no center-dup) ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dz_zernike_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --n-zcoef 5 \
    --field-x-rad 5e-5 \
    --field-y-rad 5e-5 \
    --grid 3x3 \
    --tag FFSegDemoAll_3x3 \
    --out-dir "$SCRIPT_DIR"
