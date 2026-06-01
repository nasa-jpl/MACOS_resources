#!/usr/bin/env bash
# run_dwdx.sh -- multi-field dw/dx (rigid-body) example for
# e5hex1.in.  Companion to run_dwdz.sh in this directory.
#
# Inputs the user must set per Rx:
#   --field-x-rad / --field-y-rad : test-point inset from the
#       extreme corner of the science FoV.  See run_dwdz.sh for
#       the same convention notes (additive on top of the Rx's
#       nominal ChfRayDir).
#   --dofs       : which 6-DOF rigid-body channels to perturb.
#                  Defaults to Tx,Ty,Tz here (3 DOFs * 11 optics =
#                  33 channels) to keep the demo lightweight.  Use
#                  --dofs Rx,Ry,Rz,Tx,Ty,Tz for the full 6-DOF
#                  Jacobian (66 channels, ~2x longer per field).
#
# See FFSegDemoAll/run_dwdx.sh for the full input/output catalogue.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/e5hex1.in"

echo "=== e5hex1: 5-field default (Tx,Ty,Tz only) ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dx_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --field-x-rad 1e-4 \
    --field-y-rad 1e-4 \
    --dofs Tx,Ty,Tz \
    --out-dir "$SCRIPT_DIR"
