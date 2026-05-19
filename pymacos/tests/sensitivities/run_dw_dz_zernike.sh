#!/usr/bin/env bash
# Run the dw/dz Zernike-coefficient sensitivity sweep.
#
# Thin wrapper: sources Intel oneAPI runtime, activates the pymacos
# venv, then invokes `python3 -m sensitivities.dw_dz_zernike` with any
# args passed through.
#
# Default prescription (set in dw_dz_zernike.py):
#   pymacos/tests/Rx/e5hex1.in
# (a local copy of /home/dcr/dev/macos/ZGD_test_files/e5hex1.in,
# carried in the repo so the sensitivities tests are self-contained.)
#
# Examples
#   ./run_dw_dz_zernike.sh
#       defaults: e5hex1.in, size=128, modes 4..15, kinds=monzern,zern
#   ./run_dw_dz_zernike.sh --help
#       show all CLI options
#   ./run_dw_dz_zernike.sh --model-size 256 --n-zcoef 45
#       HWO-style sweep: 378 channels on e5hex1.in
#   ./run_dw_dz_zernike.sh --rx pymacos/tests/Rx/myRx.in \
#       --kinds monzern,zern,ffzern
#       custom Rx, all three Zernike-form channel kinds
#
# Output (in ./results/):
#   dwdz_<rx_stem>.mat   -- m2v.m-compatible Jacobian + indx + w_nom
#   dwdz_<rx_stem>.png   -- panel figure of every sensitivity map

# Avoid `set -euo pipefail` here: Intel's setvars.sh exits non-zero
# under that strictness when its stdout/stderr are silenced (see also
# makejoint.sh which uses `--force` and no -e).  Explicit `-f` checks
# below cover the real failure modes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYMACOS_ROOT="$(cd "$TESTS_DIR/.." && pwd)"
VENV_ACT="$PYMACOS_ROOT/.venv/bin/activate"
ONEAPI_SETVARS="/opt/intel/oneapi/setvars.sh"

if [ ! -f "$ONEAPI_SETVARS" ]; then
    echo "** Intel oneAPI not found at $ONEAPI_SETVARS" >&2
    exit 1
fi
if [ ! -f "$VENV_ACT" ]; then
    echo "** pymacos venv not found at $VENV_ACT" >&2
    echo "   create it from $PYMACOS_ROOT before running this script" >&2
    exit 1
fi

# shellcheck disable=SC1090,SC1091
source "$ONEAPI_SETVARS" --force >/dev/null 2>&1
# shellcheck disable=SC1090
source "$VENV_ACT"

cd "$TESTS_DIR"
exec python3 -m sensitivities.dw_dz_zernike "$@"
