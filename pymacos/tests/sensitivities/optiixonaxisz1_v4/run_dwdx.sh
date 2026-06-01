#!/usr/bin/env bash
# run_dwdx.sh -- multi-field dw/dx (rigid-body) example for
# optiixonaxisz1_v4.in.  Unlike run_dwdz.sh (which exits with "no
# channels found" because optiix has no Zernike eligibility), the
# rigid-body Jacobian has plenty to perturb -- every NSReflector /
# Reflector / Reference / Return / FocalPlane element gets a 6-DOF
# perturbation channel.
#
# Required user inputs:
#   --field-x-rad / --field-y-rad  (placeholder 1e-3 -- replace with
#                                   the Rx-specific FoV when known)
#   --stop-elt                     optiix has no ApStop= in the Rx
#                                  header; Elt 10 is the EP reference
#                                  ("Reference Flat" at the system
#                                  aperture stop), so use that.
#
# Default channel setup: the full 6-DOF Jacobian (Rx, Ry, Rz, Tx,
# Ty, Tz) per actual optic.  optiix has 12 actual optics, so
# 12 × 6 = 72 channels in element order.  Add --include-non-optics
# to also perturb Reference/Return surfaces.  verifyall.m lays
# the output as N_elt rows × 6 cols.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/optiixonaxisz1_v4.in"

echo "=== optiixonaxisz1_v4: dw/dx 5-field default, 6 DOFs per element ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dx_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --field-x-rad 1e-3 \
    --field-y-rad 1e-3 \
    --stop-elt 10 \
    --out-dir "$SCRIPT_DIR"
