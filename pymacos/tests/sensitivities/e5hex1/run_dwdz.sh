#!/usr/bin/env bash
# run_dwdz.sh -- multi-field dw/dz_Zernike for e5hex1.in.
#
# Computes 42 Zernike modes per Zernike-eligible element.  Modes are
# Z4..Z45 -- we skip Z1 (piston), Z2 (x-tilt), Z3 (y-tilt) because
# they are physically redundant with the rigid-body Tz, Ry, Rx DOFs
# already captured in dw/dx.  Z4 (focus) onward is genuinely new
# information.
#
# Eligible elements are those with a ZernType / MonZernType /
# FFZernType block in the .in file -- typically the FreeForm
# primaries (MonZern channels) and any Zernike-typed elements
# (Zern channels).  --kinds defaults to monzern,zern (FFZern
# channels off by default; add 'ffzern' to --kinds if you want
# them too).
#
# State-vector layout: [Elt_a.{Z4..Z45}, Elt_b.{Z4..Z45}, ...]'
# in element order over eligible elements only.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/e5hex1.in"

echo "=== e5hex1: dw/dz 5-field default, Z4..Z45 (42 modes) per element ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dz_zernike_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --zmode-start 4 \
    --n-zcoef 45 \
    --field-x-rad 1e-4 \
    --field-y-rad 1e-4 \
    --out-dir "$SCRIPT_DIR"
