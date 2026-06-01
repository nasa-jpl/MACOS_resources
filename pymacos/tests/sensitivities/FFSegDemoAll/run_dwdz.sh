#!/usr/bin/env bash
# run_dwdz.sh -- multi-field dw/dz_Zernike for FFSegDemoAll.in.
#
# Computes 12 Zernike modes (Z4..Z15) per Zernike-eligible element
# at a 3×3 field grid.  Z1..Z3 (piston, tip, tilt) are intentionally
# skipped because they are physically redundant with the rigid-body
# Tz, Ry, Rx DOFs already captured by dw/dx.
#
# State-vector layout: [Elt_a.{Z4..Z15}, Elt_b.{Z4..Z15}, ...]'
# in element order over Z-eligible elements.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/FFSegDemoAll.in"

echo "=== FFSegDemoAll: dw/dz 3×3 grid, Z4..Z15 (12 modes) per element ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dz_zernike_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --zmode-start 4 \
    --n-zcoef 15 \
    --field-x-rad 5e-5 \
    --field-y-rad 5e-5 \
    --grid 3x3 \
    --tag FFSegDemoAll \
    --out-dir "$SCRIPT_DIR"
