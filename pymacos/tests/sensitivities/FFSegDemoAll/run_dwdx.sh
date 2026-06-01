#!/usr/bin/env bash
# run_dwdx.sh -- multi-field dw/dx (rigid-body) for FFSegDemoAll.in.
#
# Computes the full 6-DOF Jacobian (Rx, Ry, Rz, Tx, Ty, Tz) for every
# actual optic at a 3×3 field grid covering ±field-x/y rad.  State-
# vector layout:
#
#     x = [Elt1.{Rx,Ry,Rz,Tx,Ty,Tz}, Elt2.{...}, ..., EltN.{...}]'
#
# DOFs are in each element's LOCAL frame.  verifyall.m lays the
# output as N_elt rows × 6 cols (one element per row, one DOF per
# column).
#
# Required inputs:
#   --field-x-rad / --field-y-rad   half-FoV in x / y (rad)
#   --stop-obj-pos                  FFSegDemoAll has no ApStop= in
#                                   the Rx header; object-space STOP
#                                   at the global origin is the
#                                   standard 'STOP obj 0,0,0' form
#                                   and is needed for SXP-based EP
#                                   follow-up under --fp-mode=track.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/FFSegDemoAll.in"

echo "=== FFSegDemoAll: dw/dx 3×3 grid, 6 DOFs per element ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dx_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --field-x-rad 5e-5 \
    --field-y-rad 5e-5 \
    --stop-obj-pos 0,0,0 \
    --grid 3x3 \
    --tag FFSegDemoAll \
    --out-dir "$SCRIPT_DIR"
