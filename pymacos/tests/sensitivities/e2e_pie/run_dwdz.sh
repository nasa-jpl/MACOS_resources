#!/usr/bin/env bash
# run_dwdz.sh -- multi-field dw/dz_Zernike example for e2e_pie.in, the
# 7-segment pie-PM three-mirror telescope emitted by the mmacos design
# pipeline (design/examples/e2e).
#
# Unlike a bare conic telescope, e2e_pie carries Zernike (SrfType=8)
# surfaces on several downstream mirrors (elts 8, 10, 11, 12, 14), so
# the dw/dz_Zernike supervisor finds real perturbation channels and
# produces a genuine per-mode Jacobian -- a working demo of the
# Zernike-figure sensitivity path (the FreeForm PM segments carry the
# grid-figure channel exercised by dw/dgrid instead).
#
# e2e_pie declares its own object-space aperture stop (ApStop= 0 0 0),
# so no --stop-elt is needed.  4 m, 500 nm segmented TMA, model 512.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/e2e_pie.in"

echo "=== e2e_pie: dw/dz_Zernike 5-field default ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dz_zernike_multi.py \
    --rx "$RX" \
    --model-size 512 \
    --n-zcoef 5 \
    --field-x-rad 1e-3 \
    --field-y-rad 1e-3 \
    --out-dir "$SCRIPT_DIR"
