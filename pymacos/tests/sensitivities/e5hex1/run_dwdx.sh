#!/usr/bin/env bash
# run_dwdx.sh -- multi-field dw/dx (rigid-body) for e5hex1.in.
#
# Computes the full 6-DOF Jacobian (Rx, Ry, Rz, Tx, Ty, Tz) for every
# actual optic in element order.  State-vector layout:
#
#     x = [Elt1.{Rx,Ry,Rz,Tx,Ty,Tz}, Elt2.{...}, ..., EltN.{...}]'
#
# Per-element DOFs are in the element's LOCAL frame (the default
# semantic of macos's CPERTURB_PROG).  Plots in verifyall.m lay this
# out as N_elt rows × 6 columns: one element per row, one DOF per
# column.
#
# Inputs the user must set per Rx:
#   --field-x-rad / --field-y-rad : direction-cosine offsets for the
#       corner test fields, additive on top of the Rx's nominal
#       ChfRayDir.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/e5hex1.in"

echo "=== e5hex1: dw/dx 5-field default, 6 DOFs per element ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dx_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --field-x-rad 1e-4 \
    --field-y-rad 1e-4 \
    --out-dir "$SCRIPT_DIR"
