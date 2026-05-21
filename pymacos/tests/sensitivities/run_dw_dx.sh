#!/usr/bin/env bash
# Run the dw/dx rigid-body sensitivity sweep.
#
# Thin wrapper: sources Intel oneAPI runtime, activates the pymacos
# venv, then invokes `python3 -m sensitivities.dw_dx` with any args
# passed through.
#
# Default prescription (set in dw_dx.py):
#   pymacos/tests/Rx/Rx_e5hex1.in
# (local copy carried in the repo so the sensitivities tests are
# self-contained).
#
# Per-optic state vector (6 DOFs, matches GMI.F's `prb` layout):
#   (Rx, Ry, Rz, Tx, Ty, Tz)
# Element-major, DOF-minor.  Each rotation in radians; each translation
# in SI metres (converted to BaseUnits internally by pymacos.perturb).
#
# Examples
#   ./run_dw_dx.sh --rx ../Rx/Rx_e5hex1.in
#       defaults: e5hex1.in, size=128, all 6 DOFs per actual optic
#   ./run_dw_dx.sh --help
#       show all CLI options
#
#   ./run_dw_dx.sh --model-size 256
#       finer ray grid; same channel count, larger w-vector
#   ./run_dw_dx.sh --dofs Rx,Ry,Tz
#       3-DOF-per-optic sweep (rotations xy + piston only)
#   ./run_dw_dx.sh --rx pymacos/tests/Rx/myRx.in
#       custom Rx (any prescription with actual optics)
#
#   ./run_dw_dx.sh --include-source --groups auto
#       prepend 6 Src channels (perturb_src + stop re-aim) and
#       append group channels parsed from EltGrp= lines in the Rx.
#       Group perturbations route through macos GPERTURB.
#
#   ./run_dw_dx.sh --include-source --include-non-optics \
#       --groups 'all=1,2,3,4,5,6,7,8,9,10,11,12,13' \
#       --fp-mode none --group-coords global
#       Source-vs-all-optics rigid-body cross-check.  By relative
#       motion, Src{Rx,Ry} should be exactly anti-correlated with
#       Grp[all]{Rx,Ry} (cos = -1).  --include-non-optics adds
#       Reference/Return surfaces to the per-element block so the
#       saved .mat has full coverage for predict_global_rigid_response.
#
# Subtle knobs (see CLAUDE.md for full rationale):
#   --fp-mode {track,sxp,srs,fex,none}   FP perturbation handling
#   --group-coords {global,local}        GPERTURB axis frame
#   --group-stop-mode {obj,elt,none}     chief-ray re-aim after GPERTURB
#   --src-stop-mode {obj,elt,none}       chief-ray re-aim after perturb_src
#   --update-ep {none,sxp,fex}           re-derive EP before each OPD
#
# Output (in ./results/):
#   dwdx_<rx_stem>.mat   -- m2v.m-compatible Jacobian + indx + w_nom
#                           + n_src, n_elt, n_group, channel_names,
#                             group_{names,members,n_members}
#   dwdx_<rx_stem>.png   -- panel figure: rows = src(1) + optics + groups,
#                           cols = DOFs

# Avoid `set -euo pipefail`: Intel's setvars.sh exits non-zero under
# strict mode when its output is silenced; the explicit -f checks
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

# Use PYTHONPATH so the `sensitivities` package resolves WITHOUT
# cd-ing into TESTS_DIR -- otherwise relative paths the user passed
# (e.g. `--rx ../Rx/foo.in`) would resolve against the wrong cwd.
exec env PYTHONPATH="$TESTS_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m sensitivities.dw_dx "$@"
