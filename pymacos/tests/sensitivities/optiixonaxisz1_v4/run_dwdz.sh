#!/usr/bin/env bash
# run_dwdz.sh -- multi-field dw/dz_Zernike example for
# optiixonaxisz1_v4.in (on-axis 18-element optic from JPL's optiix
# regression family).
#
# NOTE: optiixonaxisz1_v4.in declares only NSReflector/Reflector/
# Reference/Return/FocalPlane elements with Conic or Flat surfaces.
# It has NO Zernike (SrfType=8), MonGrData (SrfType=13), nor FreeForm
# (SrfType=14) elements -- so the dw/dz_Zernike supervisor finds
# zero perturbation channels and exits with "no channels found".
#
# That's the EXPECTED outcome for this Rx -- it's a useful cross-
# check that the channel-discovery side correctly reports "nothing
# to perturb" instead of producing bogus sensitivities.  Treat the
# expected stderr exit as success.
#
# To exercise the dw/dz_Zernike path on an optiix-style optic, add
# a `ZernType= BornWolf` + `nZernCoef= 8` + matching `ZernModes`/
# `ZernCoef` block to one of the Reflector elements.  See
# e5hex1.in for the syntactic pattern (Elt 9 has a ZernCoef block
# on top of its Conic surface).
#
# For now this runner is documentation-only; we deliberately don't
# `set -e` so the expected SystemExit doesn't trip the wrapper.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYMACOS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PYMACOS_ROOT"

RX="$SCRIPT_DIR/optiixonaxisz1_v4.in"

echo "=== optiixonaxisz1_v4: 5-field default (expected: 'no channels found') ==="
PYTHONPATH=src:tests .venv/bin/python tests/sensitivities/dw_dz_zernike_multi.py \
    --rx "$RX" \
    --model-size 128 \
    --n-zcoef 5 \
    --field-x-rad 1e-3 \
    --field-y-rad 1e-3 \
    --out-dir "$SCRIPT_DIR" \
    || echo "(exited with 'no channels found' as expected for this Rx)"
