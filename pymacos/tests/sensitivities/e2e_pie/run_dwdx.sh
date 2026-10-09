#!/usr/bin/env bash
# run_dwdx.sh -- multi-field dw/dx (rigid-body) example for e2e_pie.in,
# the 7-segment pie-PM three-mirror telescope emitted by the mmacos
# design pipeline (templates/80_end_to_end/e2e).  Every Segment / Reflector /
# Return element gets a 6-DOF (Rx, Ry, Rz, Tx, Ty, Tz) perturbation
# channel, so the rigid-body Jacobian has plenty to perturb.
#
# e2e_pie declares its own object-space aperture stop (ApStop= 0 0 0),
# so no --stop-elt is needed; the supervisor uses the Rx's stop.
# The design is a 4 m, 500 nm segmented TMA at model size 512.
# verifyall.m lays the output as N_elt rows x 6 cols.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/e2e_pie.in"

echo "=== e2e_pie: dw/dx 5-field default, 6 DOFs per element ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dx_multi.py \
    --rx "$RX" \
    --model-size 512 \
    --field-x-rad 1e-3 \
    --field-y-rad 1e-3 \
    --out-dir "$SCRIPT_DIR"
