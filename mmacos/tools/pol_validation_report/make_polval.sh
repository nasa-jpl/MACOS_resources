#!/bin/bash
# Regenerate the polarization validation report evidence: figures, measured
# numbers, and the rendered report prose.  ONE command, per
# PLAN_POLARIZATION's "Validation document" deliverable -- no figure and no
# number in the report is ever produced by hand.
#
#   ./make_polval.sh [polvalDir]
#
# polvalDir defaults to <macos repo>/docs/macos-manual/polval, resolved from
# this script's own location (mmacos and macos are siblings under ~/dev).
#
# Two stages:
#   1. MATLAB driver run_pol_validation.m -- runs every validation case this
#      box can measure, writes media/*.png + generated/numbers.json.
#      ONE model size, ONE session (macos_init_all() corrupts the heap across
#      model_size transitions -- see mmacos/CLAUDE.md).
#   2. render_polval.py -- substitutes those numbers into polval/*.md.in ->
#      polval/*.md.  Fails if any @@TOKEN@@ is left unresolved.
#
# Then build the documents with `make polval` in docs/macos-manual.
#
# Gates this box cannot run (pymacos/ifx, GMI, the historical pre-fix engine)
# live in external.json and are labelled as externally sourced in the report,
# with the date they were captured.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMACOS="$(cd "$HERE/../.." && pwd)"
DEFAULT_POLVAL="$(cd "$MMACOS/../.." && pwd)/macos/docs/macos-manual/polval"
POLVAL="${1:-$DEFAULT_POLVAL}"

if [ ! -d "$POLVAL" ]; then
  echo "make_polval: no such directory: $POLVAL" >&2
  exit 1
fi

echo "make_polval: mmacos   = $MMACOS"
echo "make_polval: polval   = $POLVAL"

# -- stage 1: measure ---------------------------------------------------
# -batch must end with an explicit exit(0): matlab -batch hangs at implicit
# process exit once a mex has been loaded (mmacos/CLAUDE.md).
matlab -batch "run('$MMACOS/mmacos_setup.m'); \
               addpath('$HERE'); \
               run_pol_validation('$POLVAL'); \
               exit(0)"

# -- stage 2: render ----------------------------------------------------
python3 "$HERE/render_polval.py" "$POLVAL"

echo "make_polval: done -- now run 'make polval' in docs/macos-manual"
