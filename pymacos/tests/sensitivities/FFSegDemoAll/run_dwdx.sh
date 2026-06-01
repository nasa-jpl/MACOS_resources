#!/usr/bin/env bash
# run_dwdx.sh -- multi-field dw/dx (rigid-body) example for
# FFSegDemoAll.in.  Companion to run_dwdz.sh in this directory.
#
# Required user inputs:
#   --rx              path to the .in file
#   --field-x-rad     direction-cosine offset in x for corner FPs
#   --field-y-rad     direction-cosine offset in y (independent of x)
#
# Channel-setup inputs (see dw_dx.py docstring or
# dw_dx_multi.py --help for details):
#   --stop-obj-pos    FFSegDemoAll has no ApStop= in the Rx header,
#                     so the FocalPlane channel under --fp-mode=track
#                     requires an explicit stop.  Use object-space
#                     STOP at the global origin (the standard
#                     'STOP obj 0,0,0' macos invocation) -- this is
#                     re-applied per field so the new chief-ray
#                     direction is re-aimed through the same
#                     object-space stop position.
#   --dofs            channel subset; Tx,Ty,Tz here (3*8 = 24 channels)
#                     for a quick demo; switch to the full 6 DOFs for
#                     the complete 6-DOF Jacobian.
#
# Outputs land in this directory: dwdxall_FFSegDemoAll_*.mat +
# opdall_*.png + opdall_diff_*.png with tags '_5fp' / '_3x3'.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/FFSegDemoAll.in"

echo "=== FFSegDemoAll: dw/dx 5-field default (Tx,Ty,Tz) ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dx_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --field-x-rad 5e-5 \
    --field-y-rad 5e-5 \
    --stop-obj-pos 0,0,0 \
    --dofs Tx,Ty,Tz \
    --tag FFSegDemoAll_5fp \
    --out-dir "$SCRIPT_DIR"

echo
echo "=== FFSegDemoAll: dw/dx 3x3 grid (all FPs) ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dx_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --field-x-rad 5e-5 \
    --field-y-rad 5e-5 \
    --stop-obj-pos 0,0,0 \
    --dofs Tx,Ty,Tz \
    --grid 3x3 \
    --tag FFSegDemoAll_3x3 \
    --out-dir "$SCRIPT_DIR"
